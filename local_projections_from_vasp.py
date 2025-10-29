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

# Inputs
# Mesh grid. Should be commensurate with the non-self-consistent k mesh
rgrid = [ 120,120,64 ] # real-space grid for the AE wavefunctions
order = 9 # Nearest-neighbors order (1st, 2nd, 3rd, etc.)
Ecut = 10 # Planewave cut-off energy multiplicative term
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

# Other parameters
u_to_MeV = 931.49410372 # converting atomic mass to MeV/c^2
m_p = 58.6934 * u_to_MeV # Nickel mass (MeV/c^2)
m_e = 0.51099895069 # electron mass (MeV/c^2)
reduced_mass = (m_e + m_p)/m_p # (m_e+m_p)/m_e for electron and proton
Zeff = 7.5 # effecive charge Z_eff = Z - S
bohr = 0.529177*reduced_mass/Zeff # reduced Bohr radius in Angstroms

from utils import hydrogen_lattice, lattice_neighbors

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
        units = units, # Lattice vectors
        Ecut = Ecut, # Planewave cut-off energy multiplicative term
        order = order, # Nearest-neighbors order (1st, 2nd, 3rd, etc.)
        aegrid = rgrid, # real-space grid for the AE wavefunctions
    )

print('Finished')
