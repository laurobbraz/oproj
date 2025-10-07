#!/usr/bin/env python
# coding: utf-8

# # Green's function and bare susceptibility in 3D

# We use discrete Lehmann representation (DLR) imaginary frequencies to improve efficiency and decrease memory use.
# 
# Warning: the gap equation can currently only be solved when the number of fermionic and bosonic DLR frequencies is equal. Therefore, choose wmax carefully.

# In[1]:

print('Entrei')
import numpy as np
import matplotlib.pyplot as plt

# In[2]:

# main input parameters
# Units: eV                                                                                                                                                                                                                                                                                                    
wmax = 35.5 # DLR maximum imaginary frequency (larger than the band width: 3 eV)
beta = 50. # inverse temperature [200,100,50,10]
new_kgrid = [32,32,32] # k-point grid for BZ sampling

# 
# The saved data file contains the energy bands as open from the WAVECAR file.

# In[3]:


def fix_phases_along_k(c):
    """
    Fix phases of complex coefficients along k-point dimension.

    Parameters:
        c: np.ndarray of shape (Nk, Norb, Nb), complex dtype
           Projection coefficients with messy phases along k.

    Returns:
        c_fixed: np.ndarray, same shape and dtype as c,
                 phase-fixed along k dimension.
    """
    c_fixed = np.empty_like(c)
    Nk, Norb, Nb = c.shape

    # Initialize first k-point phases as is
    c_fixed[0, :, :] = c[0, :, :]

    # Loop over orbitals and bands independently
    for orb in range(Norb):
        for band in range(Nb):
            for k in range(1, Nk):
                prev = c_fixed[k-1, orb, band]
                curr = c[k, orb, band]

                # Compute phase difference
                phase_diff = np.angle(curr * np.conj(prev))

                # Remove phase difference by rotating curr
                c_fixed[k, orb, band] = curr * np.exp(-1j * phase_diff)

    return c_fixed


# In[4]:


# Loading data
psi_filename = './data/oproj_kx16ky16kz16_P.npz'
# psi_filename = './input/grid-cheio_128bands_GGA_8x8x8/oproj_128bands_GGA_kx8ky8kz8_P.npz'
file = np.load(psi_filename)
psi_kna = file['psi'] # Projected wavefunctions
nkgrid = file['nkgrid'] # k point grid
kpts_vasp = np.round(file['kpts_VASP'],9) # kpoints from TRIQS (fractional) and VASP (round needed for interpolation)
Nk = file['nk'] # # of k points
Nb = file['nb'] # # of bands
Norb = file['norb'] # # of orbitals
mu = file['mu'] # Fermi energy
eps_kn = file['eps'] # Energy bands
units = file['units'] # Lattice vectors
orb_set = file['orb_names'] # Set of orbital names (str)
r_set = file['orb_pos'] # Set of orbital positions

# Orthonormalization
from scipy.linalg import fractional_matrix_power
S_matrix = np.load( './data/overlap_matrix_oproj_kx16ky16kz16_P.npy' )
# S_matrix = np.load( './data/overlap_matrix_oproj_128bands_GGA_kx8ky8kz8_P.npy' )
S_inv_sqrt = fractional_matrix_power(S_matrix, -0.5)
psi_kna = np.einsum('ab,kbn->kan', S_inv_sqrt, psi_kna)

psi_kna = fix_phases_along_k(psi_kna)

print('Carreguei os orbitais')

# In[5]:


from scipy.interpolate import RegularGridInterpolator

# Number of points in each direction
Nx, Ny, Nz = nkgrid

# Fractional coordinates from 0 to 1 (excluding endpoint 1 for periodicity)
x = np.linspace(0, 1, Nx, endpoint=False)
y = np.linspace(0, 1, Ny, endpoint=False)
z = np.linspace(0, 1, Nz, endpoint=False)

# Mesh points (fractional coordinates)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
mesh_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

# Base interpolators (no bounds error, fill with NaN — we’ll handle periodicity)
psikan_base_interp = RegularGridInterpolator(
    (x, y, z),
    psi_kna.reshape(Nx, Ny, Nz, Norb, Nb),
    bounds_error=False, fill_value=None
)

epskn_base_interp = RegularGridInterpolator(
    (x, y, z),
    eps_kn.reshape(Nx, Ny, Nz, Nb),
    bounds_error=False, fill_value=None
)

# Periodic wrapper
def periodic_interp(interpolator):
    def wrapper(points):
        points = np.mod(points, 1.0)  # Wrap into [0,1)
        return interpolator(points)
    return wrapper

# Final periodic interpolation functions
psikan_interp = periodic_interp(psikan_base_interp)
epskn_interp = periodic_interp(epskn_base_interp)

# In[6]:


(psi_kna*psi_kna.conj()).sum(axis=(0,2))/Nk

# In[7]:


# Saving/Loading data
g_filename = "./data/g_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+"_GGA.h5"
chi0_filename = "./data/chi0_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+"_GGA.h5"

# ## Energy spectrum

# TRIQS TPRF deals with tight binding by the units, obital_positions, orbital_names, and hopping inputs.
# 
# 1. units: Lattice parameters;
# 2. orbital_positions: Orbital positions in the 'units' basis;
# 3. orbital_names: String list corresponding to each orbital position;

# In[8]:


# Defining new BZ
from triqs.lattice.tight_binding import BravaisLattice
from triqs.lattice.lattice_tools import BrillouinZone
from triqs.gf.meshes import MeshBrZone
# Lattice vectors
BL = BravaisLattice( units = units ); Vol = abs(np.linalg.det(BL.units)) # unit-cell volume
BZ = BrillouinZone( BL )
kmesh = MeshBrZone( BZ, dims=new_kgrid ); Nk = np.prod(new_kgrid)
kBZ = kmesh.bz.units # BZ unit vectors

# Getting k points from kmesh
kpts = np.array([ k for k in kmesh.values() ])
# Getting k points in fractional coordinates
inv_basis = np.linalg.inv(kBZ.T)
fractional_kpts = kpts @ inv_basis.T  # (Nk,3) matrix
kpts_TRIQS = kpts_vasp @ kBZ # kpoints from VASP in cartesian coordinates

# In[9]:


BZ

# ## Green's function

# The bare Green's function is given by
# $\begin{equation}
#     G_{a\bar{b}}(\boldsymbol{k},i\omega_n) = \sum_{m}\frac{\psi^a_{\boldsymbol{k}\nu}\psi^{\bar{b}*}_{\boldsymbol{k}\nu}}{i\omega_n+\mu-\epsilon_{\boldsymbol{k}\nu}},
# \end{equation}$
# where $\psi^a_{\boldsymbol{k}\nu}=\langle a|\phi_m\rangle$ is the projection of the $\nu$-th Kohn-Sham wavefunction $|\phi_m(\boldsymbol{k})\rangle$ over the local orbital $|a\rangle$.

# In[10]:


psikna = psikan_interp(fractional_kpts)
epskn = epskn_interp(fractional_kpts)

# In[11]:


# Constructing the imaginary-frequency Green's function
from triqs.gf import MeshDLRImFreq
wmesh = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=wmax, eps=1e-12, symmetrize=True)
ws = np.array( [ w.imag for w in wmesh ] ); nw = len(ws)
denominator = 1./(1.j*ws[:, np.newaxis, np.newaxis] + mu - epskn)
G0 = np.einsum( 'kam, kbm, wkm -> wkab', psikna, np.conj(psikna), denominator, optimize=True )
del psikna, denominator 

# In[12]:


# Constructing the TRIQS Green's function object
from triqs.gf.mesh_product import MeshProduct
from triqs.gf import Gf
Gf_mesh = MeshProduct( wmesh, kmesh ) # momentum and freq. space mesh
g0_wk = Gf(mesh=Gf_mesh, target_shape=[Norb]*2)
for nu in wmesh: # Saving Green's functions as a TRIQS Gf object
    nuii = nu.data_index
    g0_wk.data[nuii] = G0[nuii]
del G0

# Saving Green's function
from h5 import HDFArchive
with HDFArchive(g_filename, "w") as R:
    R['g0_wk'] = g0_wk
    R['beta'] = beta
    R['wmax'] = wmax
    R['nk'] = Nk
    R['nkgrid'] = nkgrid
    R['mu'] = mu

# In[13]:

print('Construi e salvei as funcoes de Green.')


# Loading Green's function
from triqs.gf import Gf
from h5 import HDFArchive
with HDFArchive(g_filename,'r') as R:
    g0_wk = R['g0_wk']

# In[14]:


g0_wk

# In[15]:


g0_wk.data.shape

# In[16]:


wmeshg0 = [ iw.imag for iw in g0_wk.mesh[0] ]
# Plot the dos with matplotlib
plt.plot( wmeshg0, g0_wk.data[:,0,0,0].real, '-' )
plt.title('$k=$'+str(kpts.reshape(Nk,3)[0]))
plt.ylabel(r'$G_{ab}(k,i\omega_m)$')
plt.xlabel(r'$i\omega_m$')
# plt.xlim(-60,60); plt.ylim(-.3,0)
# plt.xlim(-60,60)
plt.savefig('./data/gf.png')
plt.show()

# In[17]:


wmeshg0 = [ iw.imag for iw in g0_wk.mesh[0] ]
# Plot the dos with matplotlib
plt.plot( wmeshg0, g0_wk.data[:,99,0,0].real, '-' )
plt.title('$k=$'+str(kpts.reshape(Nk,3)[99]))
plt.ylabel(r'$G_{ab}(k,i\omega_m)$')
plt.xlabel(r'$i\omega_m$')
# plt.xlim(-60,60); plt.ylim(-.3,0)
# plt.xlim(-60,60)
plt.show()

print('Plotei as GF')

# ## Susceptibility

# ### Bare

# In[ ]:


# Computing the bare susceptibility
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
print('Dei load no TRIQS TPRF e comecei a calcular o chi0')
chi0_wk = imtime_bubble_chi0_wk(g0_wk, nw=1) # nw>1
del g0_wk
print('Acabei o chi0. Enfim.')

# Saving chi0
from h5 import HDFArchive
with HDFArchive(chi0_filename, "w") as R:
    R['chi0'] = chi0_wk
    R['beta'] = beta
    R['wmax'] = wmax
    R['nk'] = Nk
    R['nkgrid'] = nkgrid
    R['mu'] = mu

# In[ ]:


# In[ ]:


chi0_wk

# In[ ]:


wmeshchi0 = [ iw.imag for iw in chi0_wk.mesh[0] ]
# Plot the dos with matplotlib
kindex = 0
plt.plot( wmeshchi0, chi0_wk.data[:,kindex,0,0,0,0].real, '-' )
plt.title('$k=$'+str(kpts.reshape(Nk,3)[kindex]))
plt.ylabel(r'$\chi_{abcd}(k,i\omega_m)$')
plt.xlabel(r'$i\omega_m$')
# plt.xlim(-60,60); plt.ylim(-.3,0)
# plt.xlim(-60,60)
# plt.savefig('./data/chi0.png')
plt.show()

# In[ ]:


wmeshchi0 = [ iw.imag for iw in chi0_wk.mesh[0] ]
# Plot the dos with matplotlib
kindex = 500
plt.plot( wmeshchi0, chi0_wk.data[:,kindex,0,0,0,0].real, '-' )
plt.title('$k=$'+str(kpts.reshape(Nk,3)[kindex]))
plt.ylabel(r'$\chi_{abcd}(k,i\omega_m)$')
plt.xlabel(r'$i\omega_m$')
# plt.xlim(-60,60); plt.ylim(-.3,0)
# plt.xlim(-60,60)
plt.show()

# In[ ]:


wmeshchi0[nw//2-1]

# In[ ]:


# Static homogeneous susceptibility
nw = chi0_wk.data.shape[0]
chi0homo = (np.einsum( 'laabb->l', chi0_wk.data[nw//2] ).real).reshape(*new_kgrid)
chi0num = np.max( np.linalg.eigvals( chi0_wk.data[nw//2].reshape(Nk,Norb*Norb,Norb*Norb) ), axis=1 ).real.reshape(*new_kgrid)

# In[ ]:


# Interpolating the FFT pseudo-wavefunction from DFT
from scipy.interpolate import RegularGridInterpolator
# Number of points in each direction
Nx, Ny, Nz = new_kgrid

# Fractional coordinates from 0 to 1 (excluding endpoint 1 for periodicity)
x = np.linspace(0, 1, Nx, endpoint=False)
y = np.linspace(0, 1, Ny, endpoint=False)
z = np.linspace(0, 1, Nz, endpoint=False)

# Mesh points (fractional coordinates)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
mesh_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

chi0_base_interp = RegularGridInterpolator(
    (x, y, z),
   chi0_wk.data[nw//2].reshape(*new_kgrid,Norb,Norb,Norb,Norb),
    bounds_error=False, fill_value=None
)

# Periodic wrapper
def periodic_interp(interpolator):
    def wrapper(points):
        points = np.mod(points, 1.0)  # Wrap into [0,1)
        return interpolator(points)
    return wrapper

# Final periodic interpolation functions
chi0_interp = periodic_interp(chi0_base_interp)

# In[ ]:


# High-symmetry directions
NkHSDd = 500 # number of points along one of the HSD paths
G = [   0.,   0.,   0. ]
L = [   0.5000000000,   0.5000000000,   0.5000000000 ]
T = [   0.0000000000,   0.5000000000,   0.5000000000 ]

paths = [(G, L), (L, T), (T, G)]
NkHSD = NkHSDd*len(paths) # number of points along all the HSD paths NkHSDd*(# of paaths)

from triqs_tprf.lattice_utils import k_space_path
k_vecs, k_plot, k_ticks = k_space_path(paths, bz=BZ, num=NkHSDd)
k_vecs = k_vecs @ inv_basis.T # Converting cartesian to fractional coordinates

# Susceptibility matrix along high-symmetry directions
chi_interp = chi0_interp( k_vecs )
chi0num = np.max( np.linalg.eigvals( chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb) ),\
                                                 axis=1 ).real

# In[ ]:


plt.figure(figsize=(6,6))
plt.xticks(k_ticks, [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$'],fontsize=16)

plt.plot(k_plot, chi0num, 'k-', markersize=1)

plt.xlim( -0., np.max(k_plot) )
plt.yticks(fontsize=16)
plt.ylabel(r'$\chi^\mathrm{eig}_0(\mathbf{k},i\omega_m=0)$',fontsize=16)
# plt.title(modelname, fontsize=16)
plt.grid(True)
plt.savefig('./data/chi0.png')
print('Plotei o chi0, porra')

# ### Interacting

# In[ ]:


# The following general code was tested for an onsite multiorbital U+U'+J+J' model
U = 1.1; J = .25*U; Up = U-2*J; Jp = J
# Constructing Hamiltonian operators
# Based on the indexing ( 'spin', 'valley_orb' ),
# where 'valley_orb' first contains valleys and 
# for each valley all respective orbitals.
from triqs.operators import n, c, c_dag
from itertools import permutations
# Onsite terms
H_U = np.sum( [ 2*2*U*n(0,i)*n(1,i) + 2*2*U*n(0,i+Norb//2)*n(1,i+Norb//2) for i in range(Norb//2) ] )
delta = np.zeros( (2,2) ); i = np.arange(2); delta[i,i] = 1 # Kronecker delta
H_Up = np.sum( [ 2*2*(Up-J*delta[s,sp])*n(s,i+Norb//2)*n(sp,j) + 2*2*(Up-J*delta[s,sp])*n(s,i)*n(sp,j+Norb//2) for j in range(Norb//2) for i in range(j+1,Norb//2) for s in range(2) for sp in range(2) ] )
H_J = np.sum( [ -2*J*(c_dag(1,i+Norb//2)*c_dag(0,j)*c(1,j)*c(0,i+Norb//2)+c_dag(0,j)*c_dag(1,j)*c(0,i+Norb//2)*c(1,i+Norb//2) \
                     +c_dag(0,i+Norb//2)*c_dag(1,j)*c(0,j)*c(1,i+Norb//2)+c_dag(1,i+Norb//2)*c_dag(0,i+Norb//2)*c(1,j)*c(0,j))\
                -2*J*(c_dag(1,i)*c_dag(0,j+Norb//2)*c(1,j+Norb//2)*c(0,i)+c_dag(0,j+Norb//2)*c_dag(1,j+Norb//2)*c(0,i)*c(1,i) \
                     +c_dag(0,i)*c_dag(1,j+Norb//2)*c(0,j+Norb//2)*c(1,i)+c_dag(1,i)*c_dag(0,i)*c(1,j+Norb//2)*c(0,j+Norb//2)) for i, j in permutations(range(Norb//2), 2) ] )

H_int = H_U + H_Up + H_J
fundamental_operators = [ c(spin, orb) for spin in range(2) for orb in range(Norb) ] # indices mean ( 'spin', 'orbs' )

# Constructing quartic tensor in the Slater Hamiltonian
# from the Hamiltonian operators
from triqs_tprf.OperatorUtils import quartic_tensor_from_operator
from triqs_tprf.OperatorUtils import quartic_permutation_symmetrize
V_int_abcd = quartic_tensor_from_operator(H_int, fundamental_operators)
V_int_abcd = quartic_permutation_symmetrize(V_int_abcd)

# Reordering indices as needed for the susceptibility
V_abcd = np.zeros_like(V_int_abcd)
from itertools import product
for a, b, c, d in product(list(range(V_abcd.shape[0])), repeat=4):
    V_abcd[a, b, c, d] = V_int_abcd[b, d, a, c]

# Splitting charge and spin channels in spin space
from triqs_tprf.rpa_tensor import split_quartic_tensor_in_charge_and_spin
Uc, Us = split_quartic_tensor_in_charge_and_spin(V_abcd)

# In[ ]:


# Computing the interacting susceptibilities by hand
stoner_s = np.matmul( chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb), Us.reshape(Norb*Norb,Norb*Norb) )
stoner_c = np.matmul( chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb), -Uc.reshape(Norb*Norb,Norb*Norb) )
Id = np.eye( Norb*Norb )
chi_s = np.matmul( np.linalg.inv(Id-stoner_s), chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb) ).reshape(NkHSD,Norb,Norb,Norb,Norb)
chi_c = np.matmul( np.linalg.inv(Id-stoner_c), chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb) ).reshape(NkHSD,Norb,Norb,Norb,Norb)

# In[ ]:


# Static homogeneous interacting susceptibility
chishomo = (np.einsum( 'laabb->l', chi_s ).real)
chisnum = np.max( np.linalg.eigvals( chi_s.reshape(NkHSD,Norb*Norb,Norb*Norb) ), axis=1 ).real
chicnum = np.max( np.linalg.eigvals( chi_c.reshape(NkHSD,Norb*Norb,Norb*Norb) ), axis=1 ).real

# In[ ]:


plt.figure(figsize=(6,6))
plt.xticks(k_ticks, [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$'],fontsize=16)

plt.plot(k_plot, chisnum, 'k-', markersize=1, label=r'U='+str(U)+' eV')
#plt.scatter(k_plot, e_k_interp(k_vecs)-mu, 'ko', markersize=1)

# plt.ylim( -8, 4 )
plt.xlim( -0., np.max(k_plot) )
plt.yticks(fontsize=16)
plt.ylabel(r'$\chi^\mathrm{eig}_s(\mathbf{k},i\omega_m=0)$',fontsize=16)
# plt.title(modelname, fontsize=16)
plt.legend(fontsize=16); plt.grid(True); plt.show()

# In[ ]:


plt.figure(figsize=(6,6))
plt.xticks(k_ticks, [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$'],fontsize=16)

plt.plot(k_plot, chicnum, 'k-', markersize=1, label=r'U='+str(U)+' eV')
#plt.scatter(k_plot, e_k_interp(k_vecs)-mu, 'ko', markersize=1)

# plt.ylim( -8, 4 )
plt.xlim( -0., np.max(k_plot) )
plt.yticks(fontsize=16)
plt.ylabel(r'$\chi^\mathrm{eig}_c(\mathbf{k},i\omega_m=0)$',fontsize=16)
# plt.title(modelname, fontsize=16)
plt.legend(fontsize=16); plt.grid(True); plt.show()
