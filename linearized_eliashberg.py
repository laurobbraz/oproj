# %% [markdown]
# # Solving the linearized Eliashberg equation

# %% [markdown]
# From the TRIQS documentation.
# 
# See the theory here: https://triqs.github.io/tprf/latest/theory/eliashberg.html
# 
# The steps are
# 1. Construct the density-
# and magnetic-susceptibilties in RPA
# 2. Construct the particle-particle vertex in RPA
# 3. Construct the symmetrizing functions 
# 4. Solve the linearized Eliashberg equation

# %% [markdown]
# We use discrete Lehmann representation (DLR) imaginary frequencies to improve efficiency and decrease memory use.
# 
# Warning: the gap equation can currently only be solved when the number of fermionic and bosonic DLR frequencies is equal. Therefore, choose wmax carefully.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator # Uses linear interpolation (more trustworthy)

# %%
# main input parameters
# Units: eV
wmax = 35.5 # DLR maximum imaginary frequency (larger than the band width: 3 eV)
beta = 200 # inverse temperature [200,100,50,10]
nkgrid = (64,64,64)
nkgrid_new = (32,32,32)

# %%
# Loading data
psi_filename = './data/oproj_kx16ky16kz16_P.npz'
file = np.load(psi_filename)
nkgrid = file['nkgrid'] # k point grid
kpts_vasp = file['kpts_VASP'] # kpoints from TRIQS (fractional) and VASP
Nk = file['nk'] # # of k points
Nb = file['nb'] # # of bands
Norb = file['norb'] # # of orbitals
mu = file['mu'] # Fermi energy
units = file['units'] # Lattice vectors

# %%
# Defining new BZ
from triqs.lattice.tight_binding import BravaisLattice
from triqs.lattice.lattice_tools import BrillouinZone
from triqs.gf.meshes import MeshBrZone
# Lattice vectors
BL = BravaisLattice( units = units ); Vol = abs(np.linalg.det(BL.units)) # unit-cell volume
BZ = BrillouinZone( BL )
kmesh_new = MeshBrZone( BZ, dims=nkgrid_new )
kBZ = kmesh_new.bz.units # BZ unit vectors
inv_basis = np.linalg.inv(kBZ.T)
# Getting k points from kmesh
kpts_new = np.array([ k for k in kmesh_new.values() ])
# Getting k points in fractional coordinates
fractional_kpts_new = kpts_new @ inv_basis.T  # (Nk,3) matrix

# %%
# Saving/Loading data
g_filename = "./data/g_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+".h5"
chi0_filename = "./data/chi0_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+".h5"
delta_filename = "./data/delta_nick_"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+".h5"
deltaw0_filename = "./data/deltaw0_nick_"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+".npz"  

# %% [markdown]
# ## 1. Construct the density- and magnetic-susceptibilties in RPA

# %% [markdown]
# We first load the Green's function and the bare susceptibility.

# %%
# Loading Green's function
from triqs.gf import Gf
from h5 import HDFArchive
with HDFArchive(g_filename,'r') as R:
    g0_wk_old = R['g0_wk']
nw = g0_wk_old.data.shape[0]

# %%
# Interpolation with periodic boundary conditions
# in the new grid for chi0
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

# Creating new chi0 Gf object
from triqs.gf import Gf
from triqs.gf.mesh_product import MeshProduct
wmesh, kmesh = g0_wk_old.mesh[:]
mesh_new = MeshProduct(wmesh, kmesh_new)
g0_wk = Gf(mesh=mesh_new, target_shape=[Norb]*2)

for w in range(nw):
    g0_base_interp = RegularGridInterpolator(
        (x, y, z),
    g0_wk_old.data[w].reshape(*nkgrid,Norb,Norb),
        bounds_error=False, fill_value=None
    )

    # Periodic wrapper
    def periodic_interp(interpolator):
        def wrapper(points):
            points = np.mod(points, 1.0)  # Wrap into [0,1)
            return interpolator(points)
        return wrapper

    # Final periodic interpolation functions
    g0_interp = periodic_interp(g0_base_interp)
    g0_wk.data[w] = g0_interp( fractional_kpts_new )

del g0_wk_old

# %%
# Loading the bare susceptibility
from h5 import HDFArchive
with HDFArchive(chi0_filename,'r') as R:
    chi0_wk_old = R['chi0']
wmesh, kmesh = chi0_wk_old.mesh[:]

# %%
# Interpolation with periodic boundary conditions
# in the new grid for chi0
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

# Creating new chi0 Gf object
from triqs.gf import Gf
from triqs.gf.mesh_product import MeshProduct
wmesh, kmesh = chi0_wk_old.mesh[:]
mesh_new = MeshProduct(wmesh, kmesh_new)
chi0_wk = Gf(mesh=mesh_new, target_shape=[Norb]*4)

for w in range(nw):
    chi0_base_interp = RegularGridInterpolator(
        (x, y, z),
    chi0_wk_old.data[w].reshape(*nkgrid,Norb,Norb,Norb,Norb),
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
    chi0_wk.data[w] = chi0_interp( fractional_kpts_new )

del chi0_wk_old

# %% [markdown]
# and then solving the RPA equations for a valley-dependent Kanamori Hamiltonian, a rank 4 numpy array. We also construct the density/magentic reducible ladder-bubble vertex $\Phi^{d/m}$.

# %%
# The following general code was tested for an onsite multiorbital U+U'+J+J' model
U = 3.; J = .25*U; Up = U-2*J; Jp = J
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

# %%
# Computing the spin (magnetic) and charge (density) susceptibilities
# We reshape the rank-4 tensors to rank-2 effective two-particle
# orbitals, perform the matrix operations and then reshape back
# to the rank-4 forms.
shape = chi0_wk.data.shape # chi0 shape to reshaping

# %%
# Density channel
# Generalized Stoner criterion
stoner_d = np.matmul( chi0_wk.data.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb), -Uc.reshape(Norb*Norb,Norb*Norb) ) # density
# Charge (density) susceptibility
Id = np.eye( Norb*Norb ) # identity

chi_d_wk = np.matmul( np.linalg.inv( Id-stoner_d ), chi0_wk.data.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb) ).reshape(shape[0],shape[1],Norb,Norb,Norb,Norb) # minus sign in stoner_d
del stoner_d
# Constructing the channel-reducible
# vertex ladder and bubble functions.
# Indices are fliped using the relations
# in the documentation.
phi_d_wk = np.einsum( 'wlabcd->wlcbad', np.matmul( Uc.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_d_wk.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb), \
                                                                                      Uc.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],shape[1],Norb,Norb,Norb,Norb) )
del chi_d_wk

# %%
# Magnetic channel
# Generalized Stoner criterion
stoner_m = np.matmul( chi0_wk.data.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb), Us.reshape(Norb*Norb,Norb*Norb) ) # magnetic
# Spin (magnetic) susceptibility
Id = np.eye( Norb*Norb ) # identity
chi_m_wk = np.matmul( np.linalg.inv( Id-stoner_m ), chi0_wk.data.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb) ).reshape(shape[0],shape[1],Norb,Norb,Norb,Norb)
del stoner_m
# Constructing the channel-reducible
# vertex ladder and bubble functions.
# Indices are fliped using the relations
# in the documentation.
phi_m_wk = np.einsum( 'wlabcd->wlcbad', np.matmul( Us.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_m_wk.reshape(shape[0],shape[1],Norb*Norb,Norb*Norb), \
                                                                                      Us.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],shape[1],Norb,Norb,Norb,Norb) )
del chi_m_wk

# %% [markdown]
# ## 2. Construct the particle-particle vertex in RPA

# %% [markdown]
# Now we have all the ingredients to build the particle-particle vertex in the RPA limit.

# %%
# Constructing the singlet
# particle-particle vertex.
# The following procedure is necessary beacause of the DLR
# nonuniformly spaced frequencies.
from triqs.gf.mesh_product import MeshProduct
gamma_mesh = MeshProduct( wmesh, kmesh ) # momentum and freq. space mesh
gamma_singlet = Gf(mesh=gamma_mesh, target_shape=[Norb]*4)
for nu in wmesh:
    nuii = nu.data_index
    # Particle-particle vertex.
    gamma_singlet.data[nuii] = .5*Uc + 1.5*Us + 3.*np.conjugate( phi_m_wk[nuii] ) - np.conjugate( phi_d_wk[nuii] )
del phi_d_wk, phi_m_wk

# %% [markdown]
# ## 3. Construct the symmetrizing functions

# %% [markdown]
# By using the above vertex we must enforce the allowed SPOT symmetries of the superconducting gap $\Delta$. By using the singlet vertex we are fixing the spin symmetry to odd. We are therefore left with four physical symmetry combinations.
# 
# Spin: odd/odd
# 
# Parity (momentum): even/odd
# 
# Orbital: even/even
# 
# Time (frequency): even/odd

# %% [markdown]
# We will solve for them individually, by constructing a symmetrizing function for each of them. We do this by taking enforce_symmetry and using functools.partial to specifiy the symmetries that we want.

# %% [markdown]
# ### Orbital: Even, Frequency: Even, Momentum: Even

# %% [markdown]
# There is a problem imposing frequency symmetries when using DLR. The DLR mesh has an unequal number of elements for positive and negative frequencies so that the symmetrization function fails to compute their average.

# %%
import functools
from triqs_tprf.symmetries import enforce_symmetry
variables = ["orbital", "frequency", "momentum"]
symmetries = ["even", "even", "even"]

symmetrize_freq_even_mom_even = functools.partial(enforce_symmetry, variables=variables, symmetries=symmetries)

# %% [markdown]
# ## 4. Solve the linearized Eliashberg equation

# %% [markdown]
# Now we have everything that we need to solve the linearized Eliashberg equation. We call the solve_eliashberg function with each of our symmetrize_fcts and solve for the first leading eigenvalue, gap pair (k=1).

# %%
from triqs_tprf.eliashberg import solve_eliashberg
lambdas_freq_even_mom_even, deltas_freq_even_mom_even = solve_eliashberg(gamma_singlet, g0_wk, symmetrize_fct=symmetrize_freq_even_mom_even, k=2)

# Saving order parameter
from h5 import HDFArchive
with HDFArchive(delta_filename, "w") as R:
    R['lambda_even'] = lambdas_freq_even_mom_even
    R['delta_even'] = deltas_freq_even_mom_even
    R['lambda_odd'] = []
    R['delta_odd'] = []
    R['interactions'] = [ U, Up, J, Jp ]
    R['beta'] = beta
    R['wmax'] = wmax
    R['nk'] = Nk
    R['nkz'] = nkz
    R['mu'] = mu

# %%
# Saving only the zero frequency solution
np.savez( deltaw0_filename, 
         lambda_even=lambdas_freq_even_mom_even,
         delta_even_k1=deltas_freq_even_mom_even[0].data[nw//2],
         delta_even_k2=deltas_freq_even_mom_even[1].data[nw//2]  )

# %%
print( 'lambda_max =', lambdas_freq_even_mom_even[0])

# %% [markdown]
# Plotting them shows that all symmetries are correct and that the gap with even frequency and even momentum has the higher $\lambda$ and is therefore leading.

# %%
# Extracting data
Delta_even = np.einsum( 'wkaa->wk', deltas_freq_even_mom_even[0].data.real )
lambda_even = lambdas_freq_even_mom_even[0]

# %%
# Getting frequency points
wmeshDelta = [ iw.imag for iw in deltas_freq_even_mom_even[0].mesh[0] ]
# Getting k points from H.get_kmesh
kpts = []
for i in range(Nk):
    kpts.append( kmesh[i].value )
kpts = np.array(kpts).reshape(*nkgrid_new,3)

# %%
kindex = -1
plt.plot( wmeshDelta, Delta_even[:,kindex] )

plt.ylabel(r'$\Delta^{\mathrm{even}}(\mathbf{k}=$'+str(np.round(kpts.reshape(Nk,3)[kindex],3))+r'$,i\omega)$',fontsize=18)
plt.title(r'$\lambda^{\mathrm{even}}=$'+str(round(lambda_even,4))+r', $U=$'+str(round(U,3)),fontsize=18)
plt.xlabel(r'$i\omega$ (eV)',fontsize=18)

plt.xlim(-2,2)

plt.savefig('./data/delta_even_k'+str(kindex)+'.png', dpi=300)

plt.show()


