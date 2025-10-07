# Utility functions for the OProj data analysis

import numpy as np

def reorder_orbitals(arr, perm):
    """
    Reorder orbital indices in an array with one or more orbital axes.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (Nk, Norb, Norb, ..., Norb),
        where the first axis is k-points and the remaining are orbital indices.
    perm : list[int]
        Permutation list describing new orbital order.
        Example: [0, 2, 1, 3] for (Adx, Bdx, Adz, Bdz) -> (Adx, Adz, Bdx, Bdz).

    Returns
    -------
    np.ndarray
        Reordered array of the same shape.
    """
    if arr.ndim < 2:
        raise ValueError("Array must have at least 2 dimensions: (Nk, Norb, ...).")

    Norb = arr.shape[1]
    if not all(arr.shape[i] == Norb for i in range(1, arr.ndim)):
        raise ValueError("All orbital axes must have the same size.")
    if sorted(perm) != list(range(Norb)):
        raise ValueError("Permutation must be a rearrangement of orbital indices.")

    reordered = arr
    for ax in range(1, arr.ndim):  # apply perm to each orbital axis
        reordered = reordered.take(perm, axis=ax)
    return reordered

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


from scipy.linalg import fractional_matrix_power
def orthonormalize_projected_coeffs(psi_a_knu):
    """
    Orthonormalize projected hydrogen-like orbitals using Löwdin symmetric orthonormalization.

    Parameters
    ----------
    psi_a_knu : ndarray of shape (Nk, Norb, Nb)
        The coefficients ψ^a_{kn} = ⟨ψ_{kb} | a⟩

    Returns
    -------
    psi_a_knu_tilde : ndarray of shape (Nk, Norb, Nb)
        The orthonormalized coefficients ⟨ψ_{kn} | ψ̃^a⟩

    S : ndarray of shape (Norb, Norb)
        The original (non-orthonormal) overlap matrix
    """
    Nk, Norb, Nb = psi_a_knu.shape

    # Compute the overlap matrix S_{ab} = ∑_{kn} ψ^a_{kn}* ψ^b_{kn}
    S = np.einsum('kan,kbn->ab', psi_a_knu.conj(), psi_a_knu)/Nk

    # Löwdin orthonormalization matrix: S^{-1/2}
    S_inv_sqrt = fractional_matrix_power(S, -0.5)

    # Apply the orthonormalization to each k-point: ψ̃^a_{kn} = (S^{-1/2})_{ab} ψ^b_{kn}
    psi_a_knu_tilde = np.einsum('ab,kbn->kan', S_inv_sqrt, psi_a_knu)

    return psi_a_knu_tilde, S

def compute_orbital_resolved_spectral_function(P, eps, omega, eta=0.05, w_k=None):
    """
    Compute orbital-resolved spectral function A_a(ω) using Lorentzian broadening, with np.einsum.

    Parameters:
    -----------
    P     : ndarray, shape (Nk, Nb)
        Projection weights P_{k,a,\nu}
    eps   : ndarray, shape (Nk, Nb)
        Band energies ε_{k\nu}
    omega : ndarray, shape (Nw,)
        Energy grid ω to evaluate A(ω)
    eta   : float
        Lorentzian broadening parameter
    w_k   : ndarray, shape (Nk,), optional
        k-point weights (if None, uniform weights are used)

    Returns:
    --------
    A_omega_orb : ndarray, shape (Nw)
        Spectral function A_a(ω) resolved per orbital a
    """
    Nk, Nb = P.shape
    Nw = omega.size

    if w_k is None:
        w_k = np.ones(Nk) / Nk

    # Broadcasted energy difference (Nw, Nk, Nb)
    delta = omega[:, None, None] - eps[None, :, :]  # (Nw, Nk, Nb)

    # Lorentzian δ-function (Nw, Nk, Nb)
    lorentz = (1 / np.pi) * eta / (delta**2 + eta**2)

    # Compute: sum_kν [ w_k * P_{k a ν} * lorentz(ω - ε_{kν}) ]
    # P: (Nk, Nb), lorentz: (Nw, Nk, Nb), w_k: (Nk,)
    A_omega_orb = np.einsum('k,kb,wkb->w', w_k, P, lorentz)

    return A_omega_orb

def dos( psi_kna, eps_kn, omega_range, eta=0.05, tol=1e-3 ):
    '''
        This function computes the DOS using an
        adaptive sampling algorithm which is very
        efficient for 1D functions.
        Inputs:
            psi_kna -> array shape(Nk,Norb,Nb)
                OProj coefficients
            eps_kn -> array shape(Nk,Nb)
                Eigenenergies
            omega_range -> list shape(2)
                Minimum and maximum energy values
            eta -> float
                Broadening parameter
            tol -> float
                Tolerance for sampling algorithm
        Returns:
            omega -> list of arrays inhomogeneous shape(Norb,Nwa)
            DOS -> list of arrays inhomogeneous shape(Norb,Nwa)

    '''
    # Computing the DOS using an adaptive sampling algorithm
    # omega_range = [np.min(eps_kn), np.max(eps_kn)] # DOS frequency range
    DOS = [] # Orbitally-resolved density of states
    omega = [] # Frequency values for each orbital from adaptive sampling
    Norb = psi_kna.shape[1] # # of orbitals
    for i in range(Norb): # loop over orbitals
        def f(omega, psi_kna=psi_kna[:,i], energy=eps_kn, eta=eta, wait=True):
            # Auxiliary fuction to use in the
            # adaptive sampling algorithm.
            return compute_orbital_resolved_spectral_function((psi_kna*psi_kna.conj()).real, energy, omega, eta=eta, w_k=None)

        # Using adaptive sampling to select suitable frequencies
        omega_a, dos = sample_function(f, omega_range, tol=tol)
        omega.append(omega_a)
        DOS.append(dos[0])
    return omega, DOS

# The next two functions are from someone else, whom I could not indentify
# Thank you misterous person.
# License: Creative Commons Zero (almost public domain) http://scpyce.org/cc0
# Both functions below are used to sample the energy dependence of the 
# density of states (implemented in Fortran) in a very efficient way.
def sample_function(func, points, tol=0.05, min_points=16, max_level=16,
                    sample_transform=None):
    """
    Sample a 1D function to given tolerance by adaptive subdivision.

    The result of sampling is a set of points that, if plotted,
    produces a smooth curve with also sharp features of the function
    resolved.

    Parameters
    ----------
    func : callable
        Function func(x) of a single argument. It is assumed to be vectorized.
    points : array-like, 1D
        Initial points to sample, sorted in ascending order.
        These will determine also the bounds of sampling.
    tol : float, optional
        Tolerance to sample to. The condition is roughly that the total
        length of the curve on the (x, y) plane is computed up to this
        tolerance.
    min_point : int, optional
        Minimum number of points to sample.
    max_level : int, optional
        Maximum subdivision depth.
    sample_transform : callable, optional
        Function w = g(x, y). The x-samples are generated so that w
        is sampled.

    Returns
    -------
    x : ndarray
        X-coordinates
    y : ndarray
        Corresponding values of func(x)

    Notes
    -----
    This routine is useful in computing functions that are expensive
    to compute, and have sharp features --- it makes more sense to
    adaptively dedicate more sampling points for the sharp features
    than the smooth parts.

    Examples
    --------
    >>> def func(x):
    ...     '''Function with a sharp peak on a smooth background'''
    ...     a = 0.001
    ...     return x + a**2/(a**2 + x**2)
    ...
    >>> x, y = sample_function(func, [-1, 1], tol=1e-3)

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(-1, 1, 12000)
    >>> plt.plot(xx, func(xx), '-', x, y[0], '.')
    >>> plt.show()

    """
    return _sample_function(func, points, values=None, mask=None, depth=0,
                            tol=tol, min_points=min_points, max_level=max_level,
                            sample_transform=sample_transform)

def _sample_function(func, points, values=None, mask=None, tol=0.05,
                     depth=0, min_points=16, max_level=16,
                     sample_transform=None):
    points = np.asarray(points)

    if values is None:
        values = np.atleast_2d(func(points))

    if mask is None:
        mask = Ellipsis

    if depth > max_level:
        # recursion limit
        return points, values

    x_a = points[...,:-1][mask]
    x_b = points[...,1:][mask]

    x_c = .5*(x_a + x_b)
    y_c = np.atleast_2d(func(x_c))

    x_2 = np.r_[points, x_c]
    y_2 = np.r_['-1', values, y_c]
    j = np.argsort(x_2)

    x_2 = x_2[...,j]
    y_2 = y_2[...,j]

    # -- Determine the intervals at which refinement is necessary

    if len(x_2) < min_points:
        mask = np.ones([len(x_2)-1], dtype=bool)
    else:
        # represent the data as a path in N dimensions (scaled to unit box)
        if sample_transform is not None:
            y_2_val = sample_transform(x_2, y_2)
        else:
            y_2_val = y_2

        p = np.r_['0',
                  x_2[None,:],
                  y_2_val.real.reshape(-1, y_2_val.shape[-1]),
                  y_2_val.imag.reshape(-1, y_2_val.shape[-1])
                  ]

        sz = (p.shape[0]-1)//2

        xscale = x_2.ptp(axis=-1)
        yscale = abs(y_2_val.ptp(axis=-1)).ravel()

        p[0] /= xscale
        p[1:sz+1] /= yscale[:,None]
        p[sz+1:]  /= yscale[:,None]

        # compute the length of each line segment in the path
        dp = np.diff(p, axis=-1)
        s = np.sqrt((dp**2).sum(axis=0))
        s_tot = s.sum()

        # compute the angle between consecutive line segments
        dp /= s
        dcos = np.arccos(np.clip((dp[:,1:] * dp[:,:-1]).sum(axis=0), -1, 1))

        # determine where to subdivide: the condition is roughly that
        # the total length of the path (in the scaled data) is computed
        # to accuracy `tol`
        dp_piece = dcos * .5*(s[1:] + s[:-1])
        mask = (dp_piece > tol * s_tot)

        mask = np.r_[mask, False]
        mask[1:] |= mask[:-1].copy()


    # -- Refine, if necessary

    if mask.any():
        return _sample_function(func, x_2, y_2, mask, tol=tol, depth=depth+1,
                                min_points=min_points, max_level=max_level,
                                sample_transform=sample_transform)
    else:
        return x_2, y_2