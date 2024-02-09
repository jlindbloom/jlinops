import numpy as np
import math
from scipy.fft import dctn as sp_dctn
from scipy.fft import idctn as sp_idctn
from scipy.sparse.linalg import LinearOperator
# from scipy.sparse.linalg._interface import _CustomLinearOperator

from ..base import _CustomLinearOperator
from ..diagonal import DiagonalOperator
from ..util import get_device

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.fft import dctn as cp_dctn
    from cupyx.scipy.fft import idctn as cp_idctn

    
    
class DCT2D(_CustomLinearOperator):
    """Represents a 2-dimensional DCT transform.
    """
    def __init__(self, grid_shape, device="cpu", type=2):
        
        # Handle shape
        self.grid_shape = grid_shape
        n = math.prod(self.grid_shape)
        self.type = type
        shape = (n,n)
        
        if device == "cpu":
            
            def _matvec(x):
                return sp_dctn( x.reshape(self.grid_shape), norm="ortho", type=self.type ).flatten()
            
            def _rmatvec(x):
                return sp_idctn( x.reshape(self.grid_shape), norm="ortho", type=self.type ).flatten()
            
        else:
            
            def _matvec(x):
                return cp_dctn( x.reshape(self.grid_shape), norm="ortho", type=self.type ).flatten()
            
            def _rmatvec(x):
                return cp_idctn( x.reshape(self.grid_shape), norm="ortho", type=self.type ).flatten()
              
        super().__init__(shape, _matvec, _rmatvec, device=device)
        
    
    def to_gpu(self):
        return DCT2D(self.grid_shape, device="gpu", type=self.type)
    
    def to_cpu(self):
        return DCT2D(self.grid_shape, device="cpu", type=self.type)

    
    
def dct_get_eigvals(A, grid_shape, make_pos=False, type=2):
    """Given an SSPD LinearOperator A that is diagonalized by the 2-dimensional DCT, computes its eigenvalues.
    """
    # Shape of dct
    M, N = grid_shape
    
    device = A.device
    if device == "cpu":
        v = np.random.normal(size=(M,N)) + 10.0
        v = np.ones((M,N))
        tmp = A @ ( sp_idctn( v, norm="ortho", type=type ).flatten()  )
        tmp = tmp.reshape((M,N))
        tmp = sp_dctn( tmp, norm="ortho", type=type ).flatten()
        res = tmp/v.flatten()
        if make_pos:
            res = np.abs(res)
        return res
    else:
        v = cp.random.normal(size=(M,N)) + 10.0
        v = cp.ones((M,N))
        tmp = A @ ( cp_idctn( v, norm="ortho", type=type ).flatten()  )
        tmp = tmp.reshape((M,N))
        tmp = cp_dctn( tmp, norm="ortho", type=type ).flatten()
        res = tmp/v.flatten()
        if make_pos:
            res = cp.abs(res)
        return res



def dct_sqrt(A, grid_shape, type=2):
    """Given a LinearOperator A that is diagonalized by the 2-dimensional DCT, performs the diagonalization (computes 
    eigenvalues), computes the square root L in A = L L^T, and returns a LinearOperator representing L.
    """
    
    # Get eigenvalues
    eigvals = dct_get_eigvals(A, grid_shape, make_pos=True)
    
    # Setup 
    device = get_device(eigvals)
    P = DCT2D(grid_shape, device=device, type=type)
    sqrt_lam = DiagonalOperator( eigvals**0.5 )
    sqrt_op = P.T @ sqrt_lam
    
    return sqrt_op



def dct_pinv(A, grid_shape, eps=1e-14, type=2):
    """Given a LinearOperator A that is diagonalized by the DCT, performs the diagonalization (computes eigenvalues), returns a LinearOperator representing A^\dagger (pseudoinverse).
    """
    # Get eigenvalues
    eigvals = dct_get_eigvals(A, grid_shape, type=type)
    device = get_device(eigvals)

    # Take reciprocals of nonzero eigenvalues
    if device == "cpu":
        recip_eigvals = np.where( np.abs(eigvals) < eps, eigvals, 1.0 / np.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = np.where( np.abs(eigvals) < eps, np.zeros_like(eigvals), recip_eigvals )
    else:
        recip_eigvals = cp.where( cp.abs(eigvals) < eps, eigvals, 1.0 / cp.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = cp.where( cp.abs(eigvals) < eps, cp.zeros_like(eigvals), recip_eigvals )
    
    # DCT op
    P = DCT2D(grid_shape, device=device, type=type)
    
    # Apinv op
    Apinv = P.T @ ( DiagonalOperator(recip_eigvals) @ P)
    
    return Apinv



def dct_sqrt_pinv(A, grid_shape, eps=1e-14, type=2):
    """Given an SSPD LinearOperator A that is diagonalized by the DCT, performs the diagonalization (computes eigenvalues),
    computes the square root L in A = L L^T, and returns a LinearOperator representing L^\dagger (pseudoinverse).
    """
    # Get eigenvalues
    eigvals = dct_get_eigvals(A, grid_shape, make_pos=True, type=type)
    device = get_device(eigvals)

    # Take reciprocals of nonzero eigenvalues
    if device == "cpu":
        recip_eigvals = np.where( np.abs(eigvals) < 1e-14, eigvals, 1.0 / np.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = np.where( np.abs(eigvals) < 1e-14, np.zeros_like(eigvals), recip_eigvals )
    else:
        recip_eigvals = cp.where( cp.abs(eigvals) < 1e-14, eigvals, 1.0 / cp.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = cp.where( cp.abs(eigvals) < 1e-14, cp.zeros_like(eigvals), recip_eigvals )
    
    # DCT op
    P = DCT2D(grid_shape, device=device, type=type)
    
    # Lpinv op
    Lpinv = DiagonalOperator(recip_eigvals**0.5) @ P

    return Lpinv