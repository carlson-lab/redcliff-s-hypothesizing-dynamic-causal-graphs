"""
Code for calculating directed spectrum features.

This code is adapted from: https://github.com/neil-gallagher/directed-spectrum/
Commit: 1c0d69f
Paper: https://openreview.net/forum?id=AhlzUugOFIo

Copyright (c) 2021, Neil Gallagher
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met: 1. Redistributions of
source code must retain the above copyright notice, this list of conditions and the
following disclaimer. 2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution. 3. Neither the name
of the copyright holder nor the names of its contributors may be used to endorse or
promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from itertools import combinations
from warnings import warn
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve


DEFAULT_CSD_PARAMS = {
    "detrend": "constant",
    "window": "hann",
    "nperseg": 512,
    "noverlap": 256,
    "nfft": None,
}
"""Default parameters sent to ``scipy.signal.csd``"""


def get_directed_spectrum(
    X,
    fs,
    pairwise=True,
    max_iter=1000,
    tol=1e-6,
    csd_params={},
):
    """
    Calculate the directed spectrum from the signal ``X``.

    Parameters
    ----------
    X : numpy.ndarray
        Timeseries data from multiple channels
        Shape: ``[n_roi, time]`` or ``[n_window, n_roi, time]``
    fs : float
        Sampling frequency
    pairwise : bool, optional
        Whether to calculate the pairwise directed spectrum
    max_iter : int, optional
        Maximum number of Wilson factorization iterations
    tol : float, optional
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    f : numpy.ndarray
        Array of sample frequencies
        Shape: ``[n_freq]``
    ds : numpy.ndarray
        Directed spectrum
        Shape: ``[n_window, n_freq, n_roi, n_roi]``
    """
    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])  # [r,t] -> [1,r,t]
    assert X.ndim == 3, f"len({X.shape}) != 3"

    # Create groups list if None passed
    groups = list(range(X.shape[1]))
    group_list = np.unique(groups)
    group_idx = [[g1 == g2 for g1 in groups] for g2 in group_list]
    G = len(group_list)
    group_pairs = combinations(range(G), 2)

    # Get a cross power spectral density matrix.
    csd_params = {**DEFAULT_CSD_PARAMS, **csd_params}
    f, cpsd = csd(
        X[:, np.newaxis],
        X[:, :, np.newaxis],
        fs=fs,
        return_onesided=False,
        **csd_params,
    )  # [f], [n,r,r,f]
    cpsd = np.moveaxis(cpsd, 3, 1)  # [n,r,r,f] -> [n,f,r,r]

    if not pairwise:
        H, Sigma = _wilson_factorize(cpsd, max_iter, tol)

    ds = np.zeros((X.shape[0], csd_params["nperseg"], G, G), dtype=np.float64)
    for gp in group_pairs:
        # get indices of both groups in current pair.
        idx0 = np.array(group_idx[gp[0]])
        idx1 = np.array(group_idx[gp[1]])
        pair_idx = np.nonzero(idx0 | idx1)[0]
        sub_idx1 = idx1[pair_idx]  # subset of pair_idx in group 1

        if pairwise:
            # Get cross power spectral density matrix corresponding to ndices of
            # selected pairs.
            sub_cpsd = cpsd.take(pair_idx, axis=-2).take(pair_idx, axis=-1)

            # Factorize cross power spectral density matrix into transfer matrix (H) and
            # covariance (Sigma).
            H, Sigma = _wilson_factorize(sub_cpsd, max_iter, tol)

            ds01, ds10 = _var_to_ds(H, Sigma, sub_idx1)
        else:
            sub_H = H.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            sub_Sigma = Sigma.take(pair_idx, axis=-2).take(pair_idx, axis=-1)
            ds01, ds10 = _var_to_ds(sub_H, sub_Sigma, sub_idx1)

        # Average across channels within group.
        ds[:, :, gp[0], gp[1]] = np.diagonal(ds01, axis1=-2, axis2=-1).mean(axis=-1)
        ds[:, :, gp[1], gp[0]] = np.diagonal(ds10, axis1=-2, axis2=-1).mean(axis=-1)

    # convert to one sided spectrum
    nyquist = np.floor(len(f) / 2).astype(int)
    ds = ds[:, : (nyquist + 1)]
    ds[:, 1:nyquist] *= 2

    if len(f) % 2 != 0:
        ds[:, nyquist] *= 2
    f = np.abs(f[: (nyquist + 1)])

    return f, ds


def _wilson_factorize(cpsd, max_iter, tol, eps_multiplier=100):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma).

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.

    This code is based on an original implementation in MATLAB provided
    by M. Dhamala (mdhamala@mail.phy-ast.gsu.edu).

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.
    max_iter : int
        Max number of Wilson factorization iterations.
    tol : float
        Wilson factorization convergence tolerance value.
    eps_multiplier : int
        Constant multiplier used in stabilizing the Cholesky decomposition
        for positive semidefinite CPSD matrices.

    Returns
    -------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_signals, n_signals)
        Wilson factorization solutions for transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_signals, n_signals)
        Wilson factorization solutions for innovation covariance matrix.
    """
    cpsd_cond = np.linalg.cond(cpsd)
    if np.any(cpsd_cond > (1 / np.finfo(cpsd.dtype).eps)):
        warn("CPSD matrix is singular!")
        # Add diagonal of small values to cross-power spectral matrix to prevent it from
        # being negative semidefinite due to rounding errors
        this_eps = np.spacing(np.abs(cpsd)).max()
        cpsd = cpsd + np.eye(cpsd.shape[-1]) * this_eps * eps_multiplier

    psi, A0 = _init_psi(cpsd)
    L = cholesky(cpsd)
    H = np.zeros_like(psi)
    Sigma = np.zeros_like(A0)

    for w in range(cpsd.shape[0]):
        for i in range(max_iter):
            # These lines implement: g = psi \ cpsd / psi* + I
            psi_inv_cpsd = solve(psi[w], L[w])
            g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
            g = g + np.identity(cpsd.shape[-1])
            gplus, g0 = _plus_operator(g)

            # S is chosen so that g0 + S is upper triangular; S + S* = 0
            S = -np.tril(g0, -1)
            S = S - S.conj().transpose()
            gplus = gplus + S
            psi_prev = psi[w].copy()
            psi[w] = psi[w] @ gplus

            A0_prev = A0[w].copy()
            A0[w] = A0[w] @ (g0 + S)

            if _check_convergence(psi[w], psi_prev, tol):
                if _check_convergence(A0[w], A0_prev, tol):
                    break
        else:
            warn("Wilson factorization failed to converge.", stacklevel=2)

        # right-side solve
        H[w] = (solve(A0[w].T, psi[w].transpose(0, 2, 1))).transpose(0, 2, 1)
        Sigma[w] = A0[w] @ A0[w].T
    return (H, Sigma)


def _var_to_ds(H, Sigma, idx1):
    """Calculate directed spectrum.

    Parameters
    ----------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Cross power spectral density transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Cross power spectral density covariance matrix.
    idx1 : numpy.ndarray
        shape (n_groups,)
        Boolean mask indicating which indices are associated with group 1,
        as opposed to group 0.

    Returns
    -------
    (ds01, ds10) : tuple
        Description here.
    """
    # convert to indices from boolean
    idx0 = np.nonzero(~idx1)[0]
    idx1 = np.nonzero(idx1)[0]

    H01 = H.take(idx0, axis=-2).take(idx1, axis=-1)
    H10 = H.take(idx1, axis=-2).take(idx0, axis=-1)
    sig00 = Sigma.take(idx0, axis=-2).take(idx0, axis=-1)
    sig11 = Sigma.take(idx1, axis=-2).take(idx1, axis=-1)
    sig01 = Sigma.take(idx0, axis=-2).take(idx1, axis=-1)
    sig10 = Sigma.take(idx1, axis=-2).take(idx0, axis=-1)

    # conditional covariances
    sig1_0 = sig11 - sig10 @ solve(sig00, sig10.conj().transpose(0, 2, 1))
    sig0_1 = sig00 - sig01 @ solve(sig11, sig01.conj().transpose(0, 2, 1))

    ds10 = np.real(H01 @ sig1_0[:, np.newaxis] @ H01.conj().transpose(0, 1, 3, 2))
    ds01 = np.real(H10 @ sig0_1[:, np.newaxis] @ H10.conj().transpose(0, 1, 3, 2))
    return (ds01, ds10)


def _init_psi(cpsd):
    """Return initial psi value for wilson factorization.

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.

    Returns
    -------
    psi : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Initial value for psi used in Wilson factorization.
    h : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Initial value for A0 used in Wilson factorization.
    """
    gamma = ifft(cpsd, axis=1)
    gamma0 = gamma[:, 0]
    gamma0 = np.real((gamma0 + gamma0.conj().transpose(0, 2, 1)) / 2.0)
    h = cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, cpsd.shape[1], 1, 1)).astype(complex)
    return psi, h


def _plus_operator(g):
    """Remove all negative lag components from time-domain representation.

    Parameters
    ----------
    g: numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Frequency-domain representation to which transformation will be applied.

    Returns
    -------
    g_pos : numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Transformed version of g with negative lag components removed.
    gamma[0] : numpy.ndarray
        shape (n_groups, n_groups)
        Zero-lag component of g in time-domain.
    """
    # remove imaginary components from ifft due to rounding error.
    gamma = ifft(g, axis=0).real

    # take half of 0 lag
    gamma[0] *= 0.5

    # take half of nyquist component if fft had even # of points
    F = gamma.shape[0]
    N = np.floor(F / 2).astype(int)
    if F % 2 == 0:
        gamma[N] *= 0.5

    # zero out negative frequencies
    gamma[N + 1 :] = 0

    gp = fft(gamma, axis=0)
    return gp, gamma[0]


def _check_convergence(x, x0, tol):
    """Determine whether maximum relative change is lower than tolerance.

    Parameters
    ----------
    x : numpy.dnarray
        Current matrix/array.
    x0 : numpy.ndarray
        Previous matrix/array
    tol : float
        Tolerance value for convergence check.

    Returns
    -------
    converged : bool
        True indicates convergence has occured, False indicates otherwise.
    """
    x_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    this_eps = np.finfo(ab_x.dtype).eps
    ab_x[ab_x <= 2 * this_eps] = 1
    rel_diff = x_diff / ab_x
    converged = rel_diff.max() < tol
    return converged


if __name__ == "__main__":
    pass


###