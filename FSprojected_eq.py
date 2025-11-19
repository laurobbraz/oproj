#!/usr/bin/env python
# coding: utf-8

# # Solving the Fermi surface-projected pairing strength equation

# From the TRIQS documentation.
# 
# See the theory here: S Graser et al. New J. Phys. 11 025016 (2009).
# 
# Here, we use the exact same data as computed to solve the linearized Eliashberg equation.
# 
# The steps are
# 1. Obtain the Fermi surface points
# 2. Construct the density- and magnetic-susceptibilties in RPA at the Fermi surface
# 3. Construct the particle-particle vertex in RPA at the Fermi surface
# 4. Solve the Fermi surface-projected pairing strength equation

# We use discrete Lehmann representation (DLR) imaginary frequencies to improve efficiency and decrease memory use.
# 
# Warning: the gap equation can currently only be solved when the number of fermionic and bosonic DLR frequencies is equal. Therefore, choose wmax carefully.

# In[1]:


import os
import sys
sys.path.insert(0, os.path.abspath('./src'))
del sys
del os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# main input parameters
# Units: eV
wmax = 50. # DLR maximum imaginary frequency (larger than the band width: 3 eV)
beta = 200 # inverse temperature [200,100,50,10]
nkx = 8; nky = 8; nkz = 2 # from VASP scf
Nkx = 32; Nky = 32; Nkz = 32 # for TRIQS e to VASP nscf


# TRIQS TPRF deals with tight binding by the units, obital_positions, orbital_names, and hopping inputs.
# 
# 1. units: Lattice parameters;
# 2. orbital_positions: Orbital positions in the 'units' basis;
# 3. orbital_names: String list corresponding to each orbital position;

# The saved data file contains the energy bands as open from the WAVECAR file.

# In[3]:


# Loading data
psi_filename = './input/grid-cheio_128bands_GGA_16x16x16/oproj_kx16ky16kz16_P.npz'
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
from utils import orthonormalize_projected_coeffs
psi_kna, S_matrix = orthonormalize_projected_coeffs(psi_kna)

from utils import fix_phases_along_k, reorder_orbitals
psi_kna = fix_phases_along_k(psi_kna) # Phase shifts are so good that Psi^2 remains unchanged.
psi_kna = reorder_orbitals( psi_kna, [0,2,1,3], axes=1 )
orb_set = reorder_orbitals( orb_set, [0,2,1,3], axes=0 )


# In[4]:


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


# In[5]:


# Defining new BZ
from triqs.lattice.tight_binding import BravaisLattice
from triqs.lattice.lattice_tools import BrillouinZone
from triqs.gf.meshes import MeshBrZone
# Lattice vectors
BL = BravaisLattice( units = units ); Vol = abs(np.linalg.det(BL.units)) # unit-cell volume
BZ = BrillouinZone( BL ); VolBZ = abs(np.linalg.det(BZ.units)) # BZ volume
kmesh = MeshBrZone( BZ, dims=nkgrid )
kBZ = kmesh.bz.units # BZ unit vectors

# Getting k points from kmesh
kpts = np.array([ k for k in kmesh.values() ])
# Getting k points in fractional coordinates
inv_basis = np.linalg.inv(kBZ.T)
fractional_kpts = kpts @ inv_basis.T  # (Nk,3) matrix

BZ


# In[6]:


# Saving/Loading data
chi0_filename = "./data/chi0_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+"_GGA.h5"
delta_filename = "./data/delta_nick_mu0"+str(mu)+"_nk"+str(Nk)+"_wmax"+str(round(wmax,3))+'_beta'+str(beta)+"_GGA.npz"


# In[7]:


# Loading chi0
from h5 import HDFArchive
with HDFArchive(chi0_filename,'r') as R:
    chi0_wk = R['chi0']


# ## Susceptibility

# ### Bare

# In[8]:


# Loading chi0
chi0_wk = reorder_orbitals( chi0_wk.data[0], [0,2,1,3], axes=(1,2,3,4) )
nkchi0grid = [64,64,50]


# In[9]:


# Interpolating the FFT pseudo-wavefunction from DFT
from scipy.interpolate import RegularGridInterpolator
# Number of points in each direction
Nx, Ny, Nz = nkchi0grid

# Fractional coordinates from 0 to 1 (excluding endpoint 1 for periodicity)
x = np.linspace(0, 1, Nx, endpoint=False)
y = np.linspace(0, 1, Ny, endpoint=False)
z = np.linspace(0, 1, Nz, endpoint=False)

# Mesh points (fractional coordinates)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
mesh_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

chi0_base_interp = RegularGridInterpolator(
    (x, y, z),
   chi0_wk.reshape(*nkchi0grid,Norb,Norb,Norb,Norb),
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
del chi0_wk


# In[10]:


# High-symmetry directions
NkHSDd = 500 # number of points along one of the HSD paths
G = [   0.,   0.,   0. ]
L = [   0.5000000000,   0.5000000000,   0.5000000000 ] # X
T = [   0.00001,   0.5000000000,   0.5000000000 ] # M

paths = [(G, L), (L, T), (T, G)]
NkHSD = NkHSDd*len(paths) # number of points along all the HSD paths NkHSDd*(# of paaths)

from triqs_tprf.lattice_utils import k_space_path
k_vecs, k_plot, k_ticks = k_space_path(paths, bz=BZ, num=NkHSDd)
k_vecs = k_vecs @ inv_basis.T # Converting cartesian to fractional coordinates

# Susceptibility matrix along high-symmetry directions
chi_interp = chi0_interp( k_vecs ) # interpolated data
chi0eig = np.max( np.linalg.eigvals( chi_interp.reshape(NkHSD,Norb*Norb,Norb*Norb) ),                                                 axis=1 ).real # Main eigenvalue
chi0homo = (np.einsum( 'laabb->l', chi_interp ).real)/2 # Homogeneous susceptibility


# In[11]:


plt.figure(figsize=(6,6))
plt.xticks(k_ticks, [r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$'],fontsize=16)

plt.plot(k_plot, chi0eig, 'k-', markersize=1, label='main eigenvalue')
plt.plot(k_plot, chi0homo, 'r-', markersize=1, label='homogeneous')

plt.xlim( -0., np.max(k_plot) )
plt.yticks(fontsize=16)
plt.ylabel(r'$\chi_0(\mathbf{k},i\omega_m=0)$',fontsize=16)
plt.grid(True)
plt.legend(fontsize=14); plt.show()


# ## 1. Obtain the Fermi surface points

# We start by obtaining the Fermi surface points.

# In[12]:


# Loading FS points in 3D
kpts_FS = np.load( './data/kptsFS.npz' )['kpts'] # FS k points (3D)
kpts_FS_ind = np.load( './data/kptsFS.npz' )['indices'] # FS k point indices (3D)


# In[29]:


# OProj at the FS
i = np.arange(0,len(kpts_FS))
psikan_FS = psikan_interp( kpts_FS @ inv_basis.T )[i,:,kpts_FS_ind[i]]


# In[30]:


psi2 = (np.abs( psikan_FS )**2).sum(axis=1)


# In[15]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("", np.array([(230,230,230),(254,153,41),(215,48,31)])/255 )
norm = mpl.colors.Normalize(vmin=0., vmax=psi2.max())
sc = ax.scatter(
    kpts_FS[:, 0], kpts_FS[:, 1], kpts_FS[:, 2],
    c=psi2,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$A_\mathbf{k}(\omega=0)$", fontsize=14)

plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()


# In[32]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors
step = 80

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("", np.array([(230,230,230),(254,153,41),(215,48,31)])/255 )
norm = mpl.colors.Normalize(vmin=0., vmax=psi2.max())
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=psi2[::step],
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$A_\mathbf{k}(\omega=0)$", fontsize=14)

plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()
print(kpts_FS[::step].shape)


# In[33]:


del psikan_FS, psi2


# Contructing the scattering grid $\boldsymbol{q}=\boldsymbol{k}-\boldsymbol{k}'$ for all $\boldsymbol{k}$ and $\boldsymbol{k}'$ points in the Fermi surface.

# In[34]:


# qs matrix for a [qx,qy] grid
qs_p = kpts_FS[::step, None, :] - kpts_FS[None, ::step, :]
qs_p = qs_p.reshape(-1, kpts_FS.shape[1]) # k-k'

qs_m = kpts_FS[::step, None, :] + kpts_FS[None, ::step, :]
qs_m = qs_m.reshape(-1, kpts_FS.shape[1]) # k-(-k')
nq = len(qs_p) # # of scattering events


# In[35]:


# chatGPT-assisted
# Folding points back to the first scattering BZ
# Transforming into the BZ basis
coords_p = qs_p @ inv_basis.T  # (nq,2) matrix
del qs_p
coords_m = qs_m @ inv_basis.T  # (nq,2) matrix
del qs_m

# Fold coefficients into [0,1) interval
qsp_in = coords_p % .99 # infinitesimal displacement to avoid interpolation error
del coords_p
qsm_in = coords_m % .99 # infinitesimal displacement to avoid interpolation error
del coords_m


# ## 2. Construct the density- and magnetic-susceptibilties in RPA at the FS

# Interpolating the bare, static susceptibility ($i\nu=0$) allows computing only the Fermi surface scattering points $\boldsymbol{q}=\boldsymbol{k}-\boldsymbol{k}'$.
# 
# We then solve the RPA equations for a valley-dependent Kanamori Hamiltonian, a rank 4 numpy array.

# In[ ]:


# Computing the previously interpolated chi0 in the new set of points
i = np.arange(len(qsp_in)) # Broadcasting indices (? last index gives nan)
chi0_interp_qp = chi0_interp( np.round( qsp_in[i], 8 ) @ inv_basis.T ).reshape(nq,Norb,Norb,Norb,Norb) # computed interpolation
i = np.arange(len(qsm_in)) # Broadcasting indices (? last index gives nan)
chi0_interp_qm = chi0_interp( np.round( qsm_in[i], 8 ) @ inv_basis.T ).reshape(nq,Norb,Norb,Norb,Norb) # computed interpolation
j = np.arange(0,kpts_FS.shape[0],step=step)
chi0_interp_arr = chi0_interp( kpts_FS[j] @ inv_basis.T ).reshape(len(j),Norb,Norb,Norb,Norb) # computed interpolation


# In[ ]:


# The following general code was tested for an onsite multiorbital U+U'+J+J' model
U = 2.; J = .25*U; Up = U-2*J; Jp = J
# Constructing Hamiltonian operators
# Based on the indexing ( 'spin', 'valley_orb' ),
# where 'valley_orb' first contains valleys and 
# for each valley all respective orbitals.
from triqs.operators import n, c, c_dag
from itertools import permutations
# Onsite terms
Nsites = 2
H_U = np.sum( [ 2*2*U*n(0,i)*n(1,i) + 2*2*U*n(0,i+Nsites)*n(1,i+Nsites) for i in range(Nsites) ] )
delta = np.zeros( (2,2) ); i = np.arange(2); delta[i,i] = 1 # Kronecker delta
H_Up = np.sum( [ 2*2*(Up-J*delta[s,sp])*n(s,i)*n(sp,j) + 2*2*(Up-J*delta[s,sp])*n(s,i+Nsites)*n(sp,j+Nsites) for j in range(Nsites) for i in range(j+1,Nsites) for s in range(2) for sp in range(2) ] )
H_J = np.sum( [ -2*J*(c_dag(1,i)*c_dag(0,j)*c(1,j)*c(0,i)+c_dag(0,j)*c_dag(1,j)*c(0,i)*c(1,i)                      +c_dag(0,i)*c_dag(1,j)*c(0,j)*c(1,i)+c_dag(1,i)*c_dag(0,i)*c(1,j)*c(0,j))                -2*J*(c_dag(1,i+Nsites)*c_dag(0,j+Nsites)*c(1,j+Nsites)*c(0,i+Nsites)+c_dag(0,j+Nsites)*c_dag(1,j+Nsites)*c(0,i+Nsites)*c(1,i+Nsites)                      +c_dag(0,i+Nsites)*c_dag(1,j+Nsites)*c(0,j+Nsites)*c(1,i+Nsites)+c_dag(1,i+Nsites)*c_dag(0,i+Nsites)*c(1,j+Nsites)*c(0,j+Nsites)) for i, j in permutations(range(Nsites), 2) ] )

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


from triqs_tprf.rpa_tensor import kanamori_charge_and_spin_quartic_interaction_tensors
Ucaux, Usaux = kanamori_charge_and_spin_quartic_interaction_tensors(Nsites, U, Up, J, Jp)


# In[37]:


np.all(np.isclose(Us[:Nsites,:Nsites,:Nsites,:Nsites],Usaux))


# In[38]:


np.all(np.isclose(Us[Nsites:,Nsites:,Nsites:,Nsites:],Usaux))


# Below, check that matmul is doing the right thing. I believe matmul only multiplies the last two indices, but chi0_wk, Us and Uc are rank 4 tensors.

# In[39]:


# For plots
# Computing the spin (magnetic) and charge (density) susceptibilities
# We reshape the rank-4 tensors to rank-2 effective two-particle
# orbitals, perform the matrix operations and then reshape back
# to the rank-4 forms.
shape = chi0_interp_arr.shape # chi0 shape to reshaping
chi0_mat = chi0_interp_arr.reshape(shape[0], Norb*Norb, Norb*Norb)
Id = np.eye(Norb*Norb)

# magnetic
stoner_m = chi0_mat @ Us.reshape(Norb*Norb, Norb*Norb)
chi_m_wk = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_m_wk[i] = np.linalg.solve(Id - stoner_m[i], chi0_mat[i])
chi_m_wk = chi_m_wk.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_m


# In[40]:


stoner_d = chi0_mat @ (-Uc.reshape(Norb*Norb, Norb*Norb))
chi_d_wk = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_d_wk[i] = np.linalg.solve(Id - stoner_d[i], chi0_mat[i])
chi_d_wk = chi_d_wk.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_d, chi0_interp_arr, chi0_mat


# Plotting this over a path through the high-symmetry points looks as follows.

# In[41]:


# Static homogeneous interacting susceptibility
chimnum = np.max( np.linalg.eigvals( chi_m_wk.reshape(shape[0],Norb*Norb,Norb*Norb) ), axis=1 ).real
chidnum = np.max( np.linalg.eigvals( chi_d_wk.reshape(shape[0],Norb*Norb,Norb*Norb) ), axis=1 ).real


# In[42]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = 'Blues'
norm = mpl.colors.Normalize(vmin=0., vmax=chimnum.max())
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=chimnum,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$\chi^{\mathrm{eig}}_m(\mathbf{q},i\nu_n=0)$", fontsize=14)

plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()


# In[43]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = 'Blues'
norm = mpl.colors.Normalize(vmin=0., vmax=chidnum.max())
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=chidnum,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$\chi^{\mathrm{eig}}_d(\mathbf{q},i\nu_n=0)$", fontsize=14)

plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()


# In[44]:


del chidnum, chimnum, chi_d_wk, chi_m_wk


# In[47]:


# Computing the spin (magnetic) and charge (density) susceptibilities
# We reshape the rank-4 tensors to rank-2 effective two-particle
# orbitals, perform the matrix operations and then reshape back
# to the rank-4 forms.
shape = chi0_interp_qp.shape # chi0 shape to reshaping
chi0_mat = chi0_interp_qp.reshape(shape[0], Norb*Norb, Norb*Norb)
Id = np.eye(Norb*Norb)

# magnetic
stoner_m = chi0_mat @ Us.reshape(Norb*Norb, Norb*Norb)
chi_m_kp = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_m_kp[i] = np.linalg.solve(Id - stoner_m[i], chi0_mat[i])
# chi_m_kp = chi_m_kp.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_m
# density
stoner_d = chi0_mat @ (-Uc.reshape(Norb*Norb, Norb*Norb))
chi_d_kp = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_d_kp[i] = np.linalg.solve(Id - stoner_d[i], chi0_mat[i])
# chi_d_kp = chi_d_kp.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_d, chi0_mat, chi0_interp_qp


# In[48]:


# Computing the spin (magnetic) and charge (density) susceptibilities
# We reshape the rank-4 tensors to rank-2 effective two-particle
# orbitals, perform the matrix operations and then reshape back
# to the rank-4 forms.
shape = chi0_interp_qm.shape # chi0 shape to reshaping
chi0_mat = chi0_interp_qm.reshape(shape[0], Norb*Norb, Norb*Norb)
Id = np.eye(Norb*Norb)

# magnetic
stoner_m = chi0_mat @ Us.reshape(Norb*Norb, Norb*Norb)
chi_m_km = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_m_km[i] = np.linalg.solve(Id - stoner_m[i], chi0_mat[i])
# chi_m_km = chi_m_km.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_m
# density
stoner_d = chi0_mat @ (-Uc.reshape(Norb*Norb, Norb*Norb))
chi_d_km = np.empty_like(chi0_mat)
for i in range(shape[0]):
    chi_d_km[i] = np.linalg.solve(Id - stoner_d[i], chi0_mat[i])
# chi_d_km = chi_d_km.reshape(shape[0], Norb, Norb, Norb, Norb)
del stoner_d, chi0_mat, chi0_interp_qm


# ## 2. Construct the particle-particle vertex in RPA at the FS

# Now we have all the ingredients to build the particle-particle vertex in the RPA limit. In this example we limit us to the singlet particle-particle vertex for a symmetry constraint calculation of the Eliashberg equation.

# For generality, we will show the process which also works for multi-orbital systems, where we first construct the density/magentic reducible ladder-bubble vertex $\Phi^{d/m}$.

# In[49]:


# Constructing the channel-reducible
# vertex ladder and bubble functions.
# Indices are fliped using the relations
# in the documentation.
phi_d_kp = np.einsum( 'labcd->lcbad', np.matmul( Uc.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_d_kp,                                                  Uc.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],Norb,Norb,Norb,Norb) )
del chi_d_kp
phi_m_kp = np.einsum( 'labcd->lcbad', np.matmul( Us.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_m_kp,                                                  Us.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],Norb,Norb,Norb,Norb) )
del chi_m_kp


# In[50]:


# Constructing the channel-reducible
# vertex ladder and bubble functions.
# Indices are fliped using the relations
# in the documentation.
phi_d_km = np.einsum( 'labcd->lcbad', np.matmul( Uc.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_d_km,                                                  Uc.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],Norb,Norb,Norb,Norb) )
del chi_d_km
phi_m_km = np.einsum( 'labcd->lcbad', np.matmul( Us.reshape(Norb*Norb,Norb*Norb), np.matmul( chi_m_km,                                                  Us.reshape(Norb*Norb,Norb*Norb) ) ).reshape(shape[0],Norb,Norb,Norb,Norb) )
del chi_m_km


# And then construct the singlet particle-particle vertex.

# In[51]:


# Constructing the singlet (Graser)
# Particle-particle vertex
# in k,k' space.
gamma_singlet_p = ( np.einsum( 'abcd->cbad', (.5*Uc + .5*Us) ) + 1.5*phi_m_kp - .5*phi_d_kp ).reshape(len(kpts_FS[::step]),len(kpts_FS[::step]),Norb,Norb,Norb,Norb)
del phi_d_kp, phi_m_kp
gamma_singlet_m = ( np.einsum( 'abcd->cbad', (.5*Uc + .5*Us) ) + 1.5*phi_m_km - .5*phi_d_km ).reshape(len(kpts_FS[::step]),len(kpts_FS[::step]),Norb,Norb,Norb,Norb)
del phi_d_km, phi_m_km


# Projecting the pairing vertex onto de the band representation. We follow New J. Phys. 11 025016 (2009) and Phys. Rev. B 110, 054509 (2024) to project the orbital basis onto the band basis. 
# 
# The necessary ingredients to project the pairing vertex onto the band basis are the eigenvectors of the non-interacting Hamiltonian $P_{\boldsymbol{k}}^{\dagger}H_{0\boldsymbol{k}}P_{\boldsymbol{k}}$ such that the superconducting order parameter becomes
# $\begin{equation}
#     \Gamma^{s/t}(\boldsymbol{k},\boldsymbol{k}') = \sum_{abcd} P^*_{-\boldsymbol{k}a\nu_{-\boldsymbol{k}}}P^*_{\boldsymbol{k}b\nu_{\boldsymbol{k}}} \Gamma_{abcd}^{s/t}P_{\boldsymbol{k}'c\nu_{\boldsymbol{k}'}}P_{-\boldsymbol{k}'d\nu_{-\boldsymbol{k}'}}.
# \end{equation}$

# In[53]:


# OProj along the FS
i = np.arange(0,len(kpts_FS),step=step)
psikan_FS = psikan_interp( kpts_FS @ inv_basis.T )[i,:,kpts_FS_ind[i]] # k
psikan_FS_m = psikan_interp( -kpts_FS @ inv_basis.T )[i,:,kpts_FS_ind[i]] # -k


# In[54]:


from mpl_toolkits.mplot3d import Axes3D
ind = 0
b1, b2, b3 = kBZ  # reciprocal lattice vectors

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("", np.array([(230,230,230),(254,153,41),(215,48,31)])/255 )
norm = mpl.colors.Normalize(vmin=0., vmax=(np.abs(psikan_FS.real[:,ind])**2).max())
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=np.abs(psikan_FS.real[:,ind])**2,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$|\psi^a_{k\nu_k}|^2$", fontsize=14)

plt.title(r'$a=$'+orb_set[ind])
plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()


# In[55]:


from mpl_toolkits.mplot3d import Axes3D
ind = 1
b1, b2, b3 = kBZ  # reciprocal lattice vectors

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("", np.array([(230,230,230),(254,153,41),(215,48,31)])/255 )
norm = mpl.colors.Normalize(vmin=0., vmax=(np.abs(psikan_FS.real[:,ind])**2).max())
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=np.abs(psikan_FS.real[:,ind])**2,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$|\psi^a_{k\nu_k}|^2$", fontsize=14)

plt.title(r'$a=$'+orb_set[ind])
plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

# plt.savefig('./data_figs/FS_color.png',
#             dpi=700,
#             bbox_inches='tight',
#             pad_inches=0.02)

plt.show()


# In[56]:


# Projecting Gamma onto the Fermi surface (band basis)
Gammap_s = np.einsum( 'kd, kc, klabcd, la, lb -> kl', np.conjugate(psikan_FS_m), np.conjugate(psikan_FS), np.real(gamma_singlet_p), psikan_FS, psikan_FS_m )
del gamma_singlet_p
Gammam_s = np.einsum( 'kd, kc, klabcd, la, lb -> kl', np.conjugate(psikan_FS_m), np.conjugate(psikan_FS), np.real(gamma_singlet_m), psikan_FS_m, psikan_FS )
del gamma_singlet_m


# Then, for the singlet (triple) solution e must impose the symmetrization (antisymmetrization) condition $2\tilde{\Gamma}^{s(t)}(\boldsymbol{k},\boldsymbol{k})=\Gamma^{s(t)}(\boldsymbol{k},\boldsymbol{k})\pm\Gamma^{s(t)}(\boldsymbol{k},-\boldsymbol{k})$.

# In[57]:


# (Anti)Symmetrization
Gamma_s = 0.5*( Gammap_s + Gammam_s ); del Gammap_s, Gammam_s


# ## 4. Solve the Fermi surface-projected pairing strength equation

# Now we have everything that we need to solve the linearized Eliashberg equation. We call the solve_eliashberg function with each of our symmetrize_fcts and solve for the first leading eigenvalue, gap pair (k=1).

# In[63]:


Ak = np.abs(psikan_FS).sum(axis=1).real
kernel = -Gamma_s/len(kpts_FS[::step])*Ak
lambda_alpha, g_alpha = np.linalg.eig( kernel )
# Ordering eigenvalues and eigenvectors
index = np.real(lambda_alpha).argsort()[::-1] # indices from the highest to the lowerst
lambda_alpha = lambda_alpha[index]
g_alpha = g_alpha[:,index]


# In[64]:


print( 'Leading lambda =', lambda_alpha[0] )
# print( 'Leading lambdaA =', lambda_alphaA[0] )


# In[65]:


# Saving pairing strength equation data
np.savez( delta_filename,
          kpts_FS=kpts_FS[::step], # FS points
          kpts_FS_ind=kpts_FS_ind[::step], # FS band index
        #   vFnorm=vFnorm, # Fermi speed
        #   dSpar=dSpar, # Fermi surface area element
          Gamma_s=Gamma_s, # Pairing vertex
          lambda_alpha=lambda_alpha, # Pairing strength
          g_alpha=g_alpha, # Gap symmetry
          interactions=[U,Up,J,Jp], # Interaction strengths
          )


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors
lbda = 0  # leading eigenvalue index

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = 'RdBu'
norm = mpl.colors.Normalize(vmin=-np.max([np.abs(g_alpha[:,lbda]).min(),np.abs(g_alpha[:,lbda]).max()]), vmax=np.max([np.abs(g_alpha[:,lbda]).min(),np.abs(g_alpha[:,lbda]).max()]) )
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=g_alpha[:,lbda].real,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$\Delta_\mathbf{k}$", fontsize=14)

plt.title(r'Leading $\lambda=$'+str(round(lambda_alpha[lbda].real,4)))
plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

plt.savefig('./data/gap_leading_oproj.png',
            dpi=700,
            bbox_inches='tight',
            pad_inches=0.02)

plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
b1, b2, b3 = kBZ  # reciprocal lattice vectors
lbda = 1  # leading eigenvalue index

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# ========================
# 1) Colored scatter points
# ========================
import matplotlib as mpl
cmap = 'RdBu'
norm = mpl.colors.Normalize(vmin=-np.max([np.abs(g_alpha[:,lbda]).min(),np.abs(g_alpha[:,lbda]).max()]), vmax=np.max([np.abs(g_alpha[:,lbda]).min(),np.abs(g_alpha[:,lbda]).max()]) )
sc = ax.scatter(
    kpts_FS[::step, 0], kpts_FS[::step, 1], kpts_FS[::step, 2],
    c=g_alpha[:,lbda].real,
    cmap=cmap,
    norm=norm,
    s=8,
    alpha=1.
)

# ========================
# 2) Brillouin Zone edges
# ========================
origin = np.array([0.0, 0.0, 0.0])
corners = np.array([
    origin,
    b1,
    b2,
    b3,
    b1 + b2,
    b1 + b3,
    b2 + b3,
    b1 + b2 + b3,
])

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 4), (2, 6),
    (3, 5), (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]

for start_idx, end_idx in edges:
    line = np.vstack([corners[start_idx], corners[end_idx]])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", lw=2)

# ========================
# 3) Reciprocal lattice vectors
# ========================
for vec, name in zip([b1, b2, b3], 
                     [r"$\mathbf{b}_1$", r"$\mathbf{b}_2$", r"$\mathbf{b}_3$"]):
    ax.quiver(*origin, *vec, color="red", arrow_length_ratio=0.1, lw=3)
    ax.text(vec[0], vec[1], vec[2], name, fontsize=16, color="green", zorder=100)

# ========================
# 4) Aspect ratio & style
# ========================
all_points = np.vstack([kpts_FS, corners])
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
ranges = maxs - mins
max_range = ranges.max() / 2.0
mid = (maxs + mins) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

# Remove panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Remove axes completely
ax.set_axis_off()

# Isometric camera view
ax.view_init(elev=30+180, azim=45)

# ========================
# 5) Colorbar above
# ========================
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=-0.3, fraction=0.03)
cbar.set_label(r"$\Delta_\mathbf{k}$", fontsize=14)

plt.title(r'Subleading $\lambda=$'+str(round(lambda_alpha[lbda].real,4)))
plt.subplots_adjust(left=0, right=.5, top=.55, bottom=0)

plt.savefig('./data/gap_subleading_oproj.png',
            dpi=700,
            bbox_inches='tight',
            pad_inches=0.02)

plt.show()

