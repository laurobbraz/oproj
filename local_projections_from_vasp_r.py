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
rgrid = [ 120,120,48 ] # real-space grid for the AE wavefunctions
Ecut = 20 # Planewave cut-off energy multiplicative term
Nkx = 16; Nky = 16; Nkz = 16 # for TRIQS e to VASP nscf !!!!CHANGE HERE!!!!
psi_filename = './data/oproj_'+'kx'+str(Nkx)+'ky'+str(Nky)+'kz'+str(Nkz)+'_P.npz'
smatrix_filename = './data/overlap_matrix_'+'kx'+str(Nkx)+'ky'+str(Nky)+'kz'+str(Nkz)+'_P.npy'
psi_r_filename = './data/oproj_r_' + 'kx' + str(Nkx) + 'ky' + str(Nky) + 'kz' + str(Nkz) + '_P.npz'

# Opening raw data content for the nscf simulation
print('Loading POSCAR')
from pymatgen.core import Structure
poscar = './POSCAR'
struct = Structure.from_file(poscar)

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

### Local-orbital projections
print('Computing local projections')

# Loading data
file = np.load(psi_filename)
psi_kna = file['psi'] # projected wavefunctions
nkgrid = file['nkgrid'] # k point grid
kpts_vasp = file['kpts_VASP'] # kpoints from TRIQS (fractional) and VASP
Nk = file['nk'] # # of k points
Nb = file['nb'] # # of bands
Norb = file['norb'] # # of orbitals
mu = file['mu'] # Fermi energy
eps_kn = file['eps'] # energy bands
bohr = file['bohr']  # Bohr radius
units = file['units'] # Lattice vectors
orb_set = file['orb_names'] # Set of orbital names (str)
r_set = file['orb_pos'] # Set of orbital positions
nlm_set = file['nlm_set'] # Set of hydrogen quantum numbers (tuple)

# # Orthonormalization
# from scipy.linalg import fractional_matrix_power
# S_matrix = np.load( smatrix_filename )
# S_inv_sqrt = fractional_matrix_power(S_matrix, -0.5)
# psi_kna = np.einsum('ab,kbn->kan', S_inv_sqrt, psi_kna)

# Orthonormalization
from utils import orthonormalize_projected_coeffs
psi_kna, S_matrix = orthonormalize_projected_coeffs(psi_kna)

from mpi4py import MPI
import numpy as np

print('Computing real-space projections')

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # core index
size = comm.Get_size() # # of cores

step = 1
Nkreduced = Nk // step

# Distribute k-points among MPI ranks
kpoints_local = np.array_split(np.arange(Nkreduced), size)[rank]


def accumulate_k(k):
    """
    Compute the real-space contribution at k-point `k`.
    """
    pswfc = vaspwfc(wavecar)
    ae_wfc_k = vasp_ae_wfc_Lauro(pswfc, aecut=Ecut, poscar=poscar, potcar=potcar)

    phi_ae_n = ae_wfc_k.get_ae_wfc_k_Lauro(ikpt=k * step + 1)  # shape (Nb, Nx, Ny, Nz)
    psi_kna_k = psi_kna[k * step]  # shape (Norb, Nb)

    # Project onto real space
    psi_r = np.einsum('an,nxyz->axyz', psi_kna_k, phi_ae_n)  # (Norb, Nx, Ny, Nz)
    return psi_r


# Prepare local accumulation array
psi_a_r_local = np.zeros((Norb, *ae_wfc._aegrid), dtype='complex')

# Accumulate over local k-points only
for k in kpoints_local:
    psi_a_r_local += accumulate_k(k)

# Reduce over all ranks into root
psi_a_r = np.zeros_like(psi_a_r_local)
comm.Reduce(psi_a_r_local, psi_a_r, op=MPI.SUM, root=0)

# ----------------------------
# Save only on root process
if rank == 0:
    print("Saving data")
    np.savez(
        file=psi_r_filename,
        psi=psi_a_r,  # projected real-space wavefunctions
        nk=Nk,  # # of k points
        nkgrid=np.array([Nkx, Nky, Nkz]),  # k-point grid
        kpts_VASP=kpts_vasp,  # k-points from TRIQS and VASP
        orb_names = orb_set, # Set of orbital names (str)
        orb_pos = r_set, # Set of orbital positions
        nlm_set = nlm_set, # Set of hydrogen quantum numbers (tuple)
        nb = Nb, # # of bands
        norb = Norb, # # of orbitals
        mu = mu, # Fermi energy
        eps = epskn, # Energy bands
        bohr = bohr, # Bohr radius
        units = units # Lattice vectors
    )
    print('Finished')