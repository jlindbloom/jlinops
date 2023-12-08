import numpy as np

from scipy.linalg.lapack import dtrtri
from scipy.linalg import block_diag


def antidiag(l):
    """Returns the 'antidiag' matrix of size l x l.
    """
    return np.flip(np.eye(l), axis=0)


def naive_gsvd(A, L):
    """Computes the gsvd of the matrix pair (A, L) in a very naive and expensive way.
    Returns U, S, M, V, Xinv such that
    A = U diag( S, I ) Xinv and L = V (M 0) Xinv.
    Code assumes that leftmost dimension of A is >= its rightmost dimension, and that
    leftmost dimension of L is <= its rightmost dimension. Furthermore, it assumes that L
     has full column rank. This is the first method described in the regularization tools manual.
    """

    n = A.shape[1]
    m = A.shape[0]
    p = L.shape[0]

    assert m >= n, "must have m >= n."
    assert  p <= n, "must have p <= n."

    # Derived quantities
    o = n - p
    q = m - (n - p)

    # QR factorization of L^T and split into pieces
    Lt_qr_fac = np.linalg.qr(L.T, "complete")
    Kp = Lt_qr_fac.Q[:,:p]
    Ko = Lt_qr_fac.Q[:,p:]
    Rp = Lt_qr_fac.R[:p,:]

    # Pseudoinverse of L
    Rpinv, _ = dtrtri(Rp)
    Lpinv = Kp @ Rpinv.T

    # QR factorization of AKo and split into pieces
    AKo = A @ Ko
    AKo_qr_fac = np.linalg.qr(AKo, "complete")
    Ho = AKo_qr_fac.Q[:,:o]
    Hq = AKo_qr_fac.Q[:,o:]
    To = AKo_qr_fac.R[:o,:]

    # Rest
    Abar = Hq.T @ A @ Lpinv
    Abar_svd = np.linalg.svd(Abar, full_matrices=True)
    Ubar, Sbar, Vhbar = Abar_svd.U, Abar_svd.S, Abar_svd.Vh
    Ep = antidiag(p)

    U1 = Hq @ Ubar[:,:p] @ Ep
    U2 = Ho
    U = np.hstack([U1, U2])

    V = Vhbar.T @ Ep

    SMinv = Ep @ np.diag(Sbar) @ Ep

    d = np.diag(SMinv)
    sigmas = np.sqrt( (d**2)/(1+(d**2)))
    mus = np.sqrt(1 - (sigmas**2))

    Minv = np.diag(1/mus)
    Xinv1 = Minv @ V.T @ L 
    Xinv2 = Ho.T @ A
    Xinv = np.vstack([Xinv1, Xinv2])

    gen_svals = sigmas/mus
    S = np.diag(sigmas)
    M = np.diag(mus)

    return U, S, M, V, Xinv













