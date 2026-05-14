import numpy as np


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


