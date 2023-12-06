import numpy as np
import scipy.sparse as sps

from ..base import MatrixLinearOperator



def lanczos_bidiag(A, b, k=6):
    """Given a (m x n) LinearOperator A and a parameter k, 
    produces W, Z, B such that A W = Z B.
    Here the columns of W are an orthonormal basis for K_k(),
    Z is dense, and B is a sparse matrix with only the main diagonal
    and first lower diagonal nonzero.
    """

    # Setup
    m, n = A.shape
    B = sps.lil_matrix((k+1, k))
    W = np.zeros((n,k))
    AW = np.zeros((m,k))
    Z = np.zeros((m, k+1))
    w = np.zeros(n)
    beta = np.linalg.norm(b)
    z = b/beta
    Z[:,0] = z
    B[1,0] = beta

    # Perform bidiagonalization
    for l in range(k):
        tmp = A.rmatvec(z) - beta*w
        alpha = np.linalg.norm(tmp)
        B[l,l] = alpha
        w = (1.0/alpha)*tmp
        W[:,l] = w
        Aw = A.matvec(w)
        AW[:,l] = Aw
        tmp = Aw - alpha*z
        beta = np.linalg.norm(tmp)
        B[l+1,l] = beta
        z = (1.0/beta)*tmp
        Z[:,l+1] = z
        
    W = MatrixLinearOperator(W)
    Z = MatrixLinearOperator(Z)
    B = MatrixLinearOperator(B)
    AW = MatrixLinearOperator(AW)
        
    return W, Z, B, AW













