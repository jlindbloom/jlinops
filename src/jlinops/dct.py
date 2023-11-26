import numpy as np
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._interface import _CustomLinearOperator



def dct_diagonalized_operator_get_eigvals(A, grid_shape):
    """Given a LinearOperator A that is diagonalized by the 2-dimensional DCT, computes its eigenvalues.
    """
    M, N = grid_shape
    v = np.random.normal(size=(M,N)) + 10.0
    tmp = A @ ( idctn( v, norm="ortho" ).flatten()  )
    tmp = tmp.reshape((M,N))
    tmp = dctn( tmp, norm="ortho" ).flatten()
    res = tmp/v.flatten()
    return res



def build_dct_Lpinv(A, grid_shape):
    """Given a LinearOperator A that is diagonalized by the DCT, performs the diagonalization (computes eigenvalues),
    computes the square root L in A = L L^T, and returns a LinearOperator representing L^\dagger.
    """
    # Get eigenvalues
    eigvals = dct_diagonalized_operator_get_eigvals(A, grid_shape)
    # print(np.amin(eigvals))
    # print(np.abs(np.amin(eigvals)) < 1e-14)

    # Take reciprocals of nonzero eigenvalues
    recip_eigvals = np.where(np.abs(eigvals) < 1e-14, eigvals, 1.0 / eigvals)
    recip_eigvals = np.where(np.abs(eigvals) < 1e-14, np.zeros_like(eigvals), recip_eigvals)


    # Shape
    M, N = grid_shape

    def _matvec(x):
        x = x.reshape(grid_shape)
        tmp = dctn( x, norm="ortho" ).flatten()
        tmp = np.sqrt(recip_eigvals)*tmp
        return tmp
    
    def _rmatvec(x):
        tmp = np.sqrt(recip_eigvals)*x
        tmp = tmp.reshape(grid_shape)
        tmp = idctn( tmp, norm="ortho" ).flatten()
        return tmp

    Lpinv = LinearOperator(A.shape, matvec=_matvec, rmatvec=_rmatvec)

    return Lpinv