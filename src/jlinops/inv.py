import numpy as np

from .base import MatrixLinearOperator, _CustomLinearOperator 
from .util import issparse, tosparse
from .linalg import banded_cholesky

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import SuperLU as cp_SuperLU



class BandedCholeskyInvOperator(_CustomLinearOperator):
    """A LinearOperator representing the inverse of an input, SPD MatrixLinearOperator A.
    The inverse is computed using a banded Cholesky factorization (A should already be banded,
    no permutation will be performed).
    """

    def __init__(self, A, _superlu=None):

        assert isinstance(A, MatrixLinearOperator), "Must give MatrixOperator as an input."
        
        # Bind
        self.original_op = A
        
        # Checks
        m, n = A.shape
        assert m == n, "A must be square!"
        #assert A.A.T == A.A, "A must be square!" 
                
        # Device
        device = A.device
        
        # Enforce that underling A is a sparse type
        if not issparse(A.A):
            A = MatrixLinearOperator( tosparse(A.A).tocsc() )
        else:
            pass
        
        # Even if on GPU, factorize and make superlu object on CPU
        if device == "cpu":
            A_cpu = A
        else:
            A_cpu = A.to_cpu()
            
        # Matrix we will factorize
        mat = A_cpu.A
        
        if _superlu is None:
            
            # Perform factorization
            chol_fac, superlu = banded_cholesky( mat )
            
            # Make GPU superlu object if applicable
            if device == "gpu":
                superlu = cp_SuperLU(superlu)
                
        else: 
            superlu = _superlu
            

        # Bind superlu, A, and chol_fac
        self.superlu = superlu
        self.A = A
        self.chol_fac = chol_fac
            
        # Build matvec and rmatvec
        def _matvec(x):
            tmp = self.superlu.solve(x, trans="N")
            return tmp
        
        def _rmatvec(x):
            tmp = self.superlu.solve(x, trans="T")
            return tmp
        
        super().__init__( (n,n), _matvec, _rmatvec, device=device, dtype=A.dtype)
        
        
    def to_gpu(self):
        
        # Switch to CPU superlu
        superlu = cp_SuperLU(self.superlu)
        
        return BandedCholeskyInvOperator(self.A.to_gpu(), _superlu=superlu)
    
    
    def to_cpu(self):
        
        raise NotImplementedError




