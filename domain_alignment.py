"""
domain_alignment.py
====================
Domain shift quantification and CORAL alignment for cross-dataset EEG analysis.

Functions
---------
mmd_rbf(X, Y, gamma='auto')
    Compute MMD distance between two feature matrices.

coral_align(Xs, Xt)
    Align target Xt to source Xs using CORAL (second-order statistics).

per_subject_mmd(Xd, subjd, Xs, gamma='auto')
    Compute MMD between each DREAMER subject and full SEED-IV feature set.

align_datasets(Xd, Xs)
    Run CORAL and return aligned versions of both datasets.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


# ─────────────────────────────────────────────────────────────────────────────
# 1. MMD (Maximum Mean Discrepancy) with RBF kernel
# ─────────────────────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma='auto') -> float:
    """
    Compute unbiased MMD^2 between X and Y using RBF kernel.

    Parameters
    ----------
    X, Y  : (n, d) and (m, d) feature matrices
    gamma : RBF kernel bandwidth. 'auto' uses median heuristic.

    Returns
    -------
    float : MMD^2 value (≥ 0; larger = more domain shift)
    """
    if gamma == 'auto':
        # Median heuristic on pooled pairwise distances
        Z = np.vstack([X, Y])
        dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_dist + 1e-8)

    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    n, m = len(X), len(Y)
    # Unbiased estimator: remove diagonal
    np.fill_diagonal(XX, 0)
    np.fill_diagonal(YY, 0)
    mmd2 = (XX.sum() / (n * (n - 1))
            + YY.sum() / (m * (m - 1))
            - 2 * XY.mean())
    return float(max(mmd2, 0.0))   # clip numerical negatives


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORAL alignment
# ─────────────────────────────────────────────────────────────────────────────

def coral_align(Xs: np.ndarray, Xt: np.ndarray,
                reg: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """
    CORAL: align Xt to Xs by matching second-order statistics (covariance).

    Both datasets are whitened (zero covariance → identity), then
    re-colored with the source covariance. After alignment, Xt_aligned
    has the same covariance structure as Xs.

    Parameters
    ----------
    Xs  : (ns, d) source features  (SEED-IV)
    Xt  : (nt, d) target features  (DREAMER)
    reg : regularization added to covariance diagonal

    Returns
    -------
    Xs_aligned, Xt_aligned : both in the same aligned space
    """
    d = Xs.shape[1]

    # Center
    mu_s = Xs.mean(axis=0)
    mu_t = Xt.mean(axis=0)
    Xs_c = Xs - mu_s
    Xt_c = Xt - mu_t

    # Covariance + regularization
    Cs = np.cov(Xs_c.T) + reg * np.eye(d)
    Ct = np.cov(Xt_c.T) + reg * np.eye(d)

    # Cholesky decomposition: C = L @ L.T
    Ls = np.linalg.cholesky(Cs)   # source whitening matrix
    Lt = np.linalg.cholesky(Ct)   # target whitening matrix

    # Whiten both → re-color with source covariance
    # Xt_aligned = Xt_white @ Ls.T  (re-colored to source distribution)
    Xs_white = Xs_c @ np.linalg.inv(Ls.T)
    Xt_white = Xt_c @ np.linalg.inv(Lt.T)

    # Re-color both with source covariance so they share the same space
    Xs_aligned = Xs_white @ Ls.T + mu_s
    Xt_aligned = Xt_white @ Ls.T + mu_s   # target re-colored to source

    return Xs_aligned, Xt_aligned


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-subject MMD
# ─────────────────────────────────────────────────────────────────────────────

def per_subject_mmd(Xd: np.ndarray, subjd: np.ndarray,
                    Xs: np.ndarray,
                    gamma='auto') -> dict[int, float]:
    """
    Compute MMD between each DREAMER subject and the full SEED-IV set.

    Parameters
    ----------
    Xd    : (nd, d) DREAMER features (raw, before alignment)
    subjd : (nd,)  subject ID per DREAMER sample
    Xs    : (ns, d) SEED-IV features
    gamma : RBF kernel bandwidth

    Returns
    -------
    dict {subject_id: mmd_value}
    """
    # Compute gamma once on full pool for consistency
    if gamma == 'auto':
        Z = np.vstack([Xd, Xs])
        dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_dist + 1e-8)

    results = {}
    for subj in np.unique(subjd):
        Xd_subj = Xd[subjd == subj]
        if len(Xd_subj) < 2:
            results[int(subj)] = float('nan')
            continue
        results[int(subj)] = mmd_rbf(Xd_subj, Xs, gamma=gamma)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def align_datasets(Xd: np.ndarray, Xs: np.ndarray,
                   reg: float = 1e-5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper: run CORAL alignment on full datasets.

    Returns
    -------
    Xs_aligned, Xt_aligned
    """
    return coral_align(Xs, Xd, reg=reg)
