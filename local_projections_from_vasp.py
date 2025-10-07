'''
    This program 

    1) Takes input from the self-consistent DFT simulation, given by VASP as the POSCAR output, to read the relaxed lattice parameters and generate a momentum $k$ grid in the Brillouin zone. This grid should be further used to perform the non-self-consistent DFT run; 

    2) Reads the WAVECAR and computes the pseudo-wavefunctions in real space [Phys. Rev. B 87, 041406(R) (2013)];

    3) Projects the pseudo-wavefunctions onto local orbitals centered in a given atomic position.

    See documentation for more details about the WAVECAR (https://www.vasp.at/wiki/index.php/WAVECAR) and POSCAR (https://www.vasp.at/wiki/index.php/POSCAR).

    See more about pymatgen and its interfce with VASP in https://pymatgen.org/pymatgen.io.vasp.html.

    TRIQS is used to generate suitable $k$ points for the many-body simulations.
    This program assumes it runs in the directory containing VASP output files.
'''
import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
del sys
del os
import numpy as np
import sphericart

# Inputs
# Mesh grid. Should be commensurate with the non-self-consistent k mesh
rgrid = [ 120,120,64 ] # real-space grid for the AE wavefunctions
order = 9 # Nearest-neighbors order (1st, 2nd, 3rd, etc.)
Ecut = 20 # Planewave cut-off energy multiplicative term
Nkx = 16; Nky = 16; Nkz = 16 # for TRIQS e to VASP nscf !!!!CHANGE HERE!!!!
psi_filename = './data/oproj_'+'kx'+str(Nkx)+'ky'+str(Nky)+'kz'+str(Nkz)+'_P.npz'
# psi_filename = './data/oproj_'+'kx'+str(Nkx)+'ky'+str(Nky)+'kz'+str(Nkz)+'_P_HSD.npz'

# Opening raw data content for the nscf simulation
print('Loading POSCAR')
from pymatgen.core import Structure
poscar = './POSCAR'
struct = Structure.from_file(poscar)
# Extracting DFT lattice parameters
units = struct.lattice.matrix # lattice basis vectors

# Loading WAVECAR. (It may take a while for large WAVECARs)
print('Loading WAVECAR')
from vaspwfc import vaspwfc
wavecar = './WAVECAR'
pswfc = vaspwfc(wavecar) # VASP pseudo-wavefunction

# All-electron wavefunction
from aewfc import vasp_ae_wfc_Lauro
potcar = './POTCAR'
ae_wfc = vasp_ae_wfc_Lauro(pswfc, aecut=Ecut, poscar=poscar, potcar=potcar, aegrid=rgrid)

# First, quick run to set up the real-space grid parameters
# Single band and k point
kpoint = 1; band = 1
phi_ae = ae_wfc.get_ae_wfc_Lauro(ikpt=kpoint, iband=band); del phi_ae
print('Working!')
# k-points, energies, and constants
# G-point dependent quantities need an explicit loop over k
Nk = pswfc._nkpts # # of k points
Nb = pswfc._nbands # # of bands
kpts_vasp = np.array(pswfc._kvecs) # k points
epskn = np.array( pswfc.readWFBand()[1][0] ) # energies shape(Nk,Nb)
mu = pswfc._efermi # chemical potential
print('Nk =', Nk)

### Creating the local orbital grid
# Setting a notation for the atomic positions
NiA = struct.sites[3].frac_coords @ units
OA1 = struct.sites[6].frac_coords @ units
OA2 = struct.sites[9].frac_coords @ units
NiB = struct.sites[4].frac_coords @ units
OB1 = struct.sites[11].frac_coords @ units
OB2 = struct.sites[8].frac_coords @ units
Oz = struct.sites[5].frac_coords @ units

Ni_pos = np.array( [ NiA, NiB ] ) # Nickel atomic postitions (A, B)
O_pos = np.array( [ OA1, OA2, OB1, OB2, Oz ] ) # Oxygen atomic postitions
rNi = np.linalg.norm(Ni_pos,axis=1) # Length of each atomic position from the origin
rO = np.linalg.norm(O_pos,axis=1) # Length of each atomic position from the origin

# Input parameters for the local-orbital projections
Norb = 4 # # number of orbitals

# Hydrogen orbitals. Same convention as Wannier90 and VASP LOCPROJ
nlm_set = np.zeros( (Norb,3), dtype=int ) # set of hydrogen quantum-number indices
r_a_set = np.zeros( (Norb,3) ) # set of fractional spacial positions for each orbital
orb_set = np.zeros( (Norb), dtype='<U16' ) # set of orbital names. Will be saved
# d_{x^2-y^2} A
orb_set[0] = r'$Ad_{x^2-y^2}$'
n = 3; l = 2; m = -2+l
nlm_set[0] = (n,l,m)
r_a_set[0] = NiA # Ni site on layer A
# d_{x^2-y^2} B
orb_set[1] = r'$Bd_{x^2-y^2}$'
n = 3; l = 2; m = -2+l
nlm_set[1] = (n,l,m)
r_a_set[1] = NiB # Ni site on layer B
# d_{3z^2-r^2} A
orb_set[2] = r'$Ad_{3z^2-r^2}$'
n = 3; l = 2; m = 0+l
nlm_set[2] = (n,l,m)
r_a_set[2] = NiA # Ni site on layer A
# d_{3z^2-r^2} B
orb_set[3] = r'$Bd_{3z^2-r^2}$'
n = 3; l = 2; m = 0+l
nlm_set[3] = (n,l,m)
r_a_set[3] = NiB # Ni site on layer B

from scipy.special import genlaguerre, factorial
# Other parameters
u_to_MeV = 931.49410372 # converting atomic mass to MeV/c^2
m_p = 58.6934 * u_to_MeV # Nickel mass (MeV/c^2)
m_e = 0.51099895069 # electron mass (MeV/c^2)
reduced_mass = (m_e + m_p)/m_p # (m_e+m_p)/m_e for electron and proton
Zeff = 7.5 # effecive charge Z_eff = Z - S
bohr = 0.529177*reduced_mass/Zeff # reduced Bohr radius in Angstroms

def spherical_coords(xyz):
    # Transforming from cartesian 
    # to spherical coordinates.
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[...,0]**2 + xyz[...,1]**2
    ptsnew[...,0] = np.sqrt(xy + xyz[...,2]**2) # r
    ptsnew[...,1] = np.arctan2(np.sqrt(xy), xyz[...,2]) # for elevation angle defined from Z-axis down (theta)
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[...,2] = np.arctan2(xyz[...,1], xyz[...,0]) # phi
    return ptsnew # [r, theta, phi]

def spherical_harmonics( l, mr, xyz, epsilon=1e-7 ):
    # Computes all the (l,m) elements
    # ranging through m first and then l
    sh = sphericart.SphericalHarmonics(l_max=l) # setting up harmonics
    r = np.linalg.norm(xyz,axis=1) # spherical coordinate r
    xyz[ r < epsilon ] = epsilon # avoiding the origin
    sh_values = sh.compute(xyz) # proper computation
    last_m = sh_values.shape[1] - (2*l+1) # last (2l+1) elements
    return sh_values[:,last_m:][:,mr]

def hydrogem_atom(rtp, xyz, nlm, bohr):
    # Contructing hydrogen-like a_nlm functions
    r, theta, phi = rtp.T # Spherical coordinates
    n, l, mr = nlm # Quantum numbers
    rho = 2*r/n/bohr # Effective radius
    Ylm = spherical_harmonics( l, mr, xyz ) # Spherical harmonics
    Llm = genlaguerre(n - l - 1, 2 * l + 1)(rho) # Generalized Laguerre polynomials
    a_nlm_coeffs = np.sqrt( np.power(2./n/bohr, 3)*( factorial(n-l-1)/factorial(n+l)/2./n ) ) # Other a_nlm coefficients
    a_nlm_coeffs *= np.exp(-rho/2.)
    a_nlm_coeffs *= np.power(rho,l)
    
    return a_nlm_coeffs*Llm*Ylm # shape( rtp.shape[0] )

def hydrogen_lattice(nlm, atom_positions, points, ngrid, bohr, max_workers=None):
    """
    Efficient multithreaded hydrogenic projection avoiding memory overload.
    Parallelizes over N_cells with online accumulation.
    By chatGPT based on my code.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    points = np.asarray(points)
    atom_positions = np.asarray(atom_positions)
    N_cells, Norb = atom_positions.shape[:2]
    N_points = points.shape[0]

    orbitals = np.zeros((Norb, N_points), dtype='complex')  # shared accumulator
    lock = Lock()

    def worker(cell_idx):
        partial = np.zeros((Norb, N_points), dtype='complex')
        center = atom_positions[cell_idx]
        for i in range(Norb):
            R = points - center[i]
            rtp = spherical_coords(R)
            partial[i] = hydrogem_atom(rtp, R, nlm[i], bohr)
        with lock:
            orbitals[:] += partial

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(worker, range(N_cells))

    # return orbitals.reshape((Norb, *ngrid)) / np.sqrt(N_cells)
    # Enforcing unit-cell normalization
    return ( orbitals / np.sqrt(np.sum( orbitals*orbitals.conj(), axis=1 )[:,None] )).reshape((Norb, *ngrid))

def lattice_neighbors(latt_vec, order=1, exclude_origin=False, include_distance=False):
    from itertools import product
    """
    Generate all lattice vectors up to the given order of neighbors.

    Parameters:
        R1, R2, R3          : numpy arrays of shape (3,), primitive lattice vectors
        order               : int, maximum sum of |n1| + |n2| + |n3| for neighbors
        exclude_origin      : bool, if True, removes the (0,0,0) vector
        include_distance    : bool, if True, returns (vector, distance) pairs

    Returns:
        neighbors           : list of numpy arrays of shape (3,) or (3,), distance
    """
    neighbors = []
    R1, R2, R3, = latt_vec.T

    # Consider all integer triplets within the cube [-order, order]
    for n1, n2, n3 in product(range(-order, order+1), repeat=3):
        if exclude_origin and (n1 == 0 and n2 == 0 and n3 == 0):
            continue
        manhattan = abs(n1) + abs(n2) + abs(n3)
        if manhattan <= order:
            vec = n1 * R1 + n2 * R2 + n3 * R3
            if include_distance:
                dist = np.linalg.norm(vec)
                neighbors.append((vec, dist))
            else:
                neighbors.append(vec)

    return np.array( neighbors )

# Getting the fractional real-space coordinates from the FFT
rx_fractional = np.fft.fftfreq(ae_wfc._aegrid[0]) % 1.
ry_fractional = np.fft.fftfreq(ae_wfc._aegrid[1]) % 1.
rz_fractional = np.fft.fftfreq(ae_wfc._aegrid[2]) % 1.
rmesh_fractional = np.array( [ [ rx, ry, rz ] \
                               for rx in rx_fractional \
                               for ry in ry_fractional \
                               for rz in rz_fractional ] )

# Fractional to Cartesian transformation
rmesh = rmesh_fractional @ units # Fractional to Cartesian coordinates
units_inv = np.linalg.inv( units.T ) # Useful Cartesian-to-Fractional converter
del rx_fractional, ry_fractional, rz_fractional, rmesh_fractional

print('Computing spherical harmonics')
# Atom positions
neighbors = lattice_neighbors(units, order=order)
atom_positions = (r_a_set+neighbors[:,None,:]) # shape( NN, Norb, 3 )
print('Working!')
# Evaluate 2p_z orbitals at all points
a_nlm = hydrogen_lattice(nlm_set,
                        atom_positions=atom_positions,
                        points=rmesh,
                        ngrid=ae_wfc._aegrid,
                        bohr=bohr)

### Local-orbital projections
print('Computing local projections')

from mpi4py import MPI
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

step = 1
Nkreduced = Nk // step

# Distribute k-points
kpoints_local = np.array_split(np.arange(Nkreduced), size)[rank]

def compute_projection_k(k):
    pswfc = vaspwfc(wavecar)
    ae_wfc = vasp_ae_wfc_Lauro(pswfc, aecut=Ecut, poscar=poscar, potcar=potcar, aegrid=rgrid)
    phi_ae_n = ae_wfc.get_ae_wfc_k_Lauro(ikpt=k * step + 1)
    proj = np.einsum('nxyz,axyz->an', phi_ae_n.conj(), a_nlm)
    return proj

# Initialize local array with zeros (full size, sparse fill)
psi_local_sum = np.zeros((Nkreduced, Norb, Nb), dtype='complex')

# Compute projections only for local k-points
for k in kpoints_local:
    proj = compute_projection_k(k)
    psi_local_sum[k] = proj

# Reduce (sum) into root process
psi_kna = np.zeros_like(psi_local_sum)
comm.Reduce(psi_local_sum, psi_kna, op=MPI.SUM, root=0)

# Only root holds the final result
if rank == 0:
    print("Done computing local projections.")
    # psi_kna now contains all the projections at root

    print('Saving data')
    # Saving data to npz format
    np.savez( 
        file=psi_filename,
        psi = psi_kna, # Projected wavefunctions
        nk = Nk, # # of k points
        nkgrid = np.array([ Nkx, Nky, Nkz ]), # k point grid
        kpts_VASP = kpts_vasp, # kpoints from VASP
        orb_names = orb_set, # Set of orbital names (str)
        orb_pos = r_a_set, # Set of orbital positions
        nlm_set = nlm_set, # Set of hydrogen quantum numbers (tuple)
        nb = Nb, # # of bands
        norb = Norb, # # of orbitals
        mu = mu, # Fermi energy
        eps = epskn, # Energy bands
        bohr = bohr, # Bohr radius
        units = units # Lattice vectors
    )

print('Finished')
