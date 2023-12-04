import numpy as np

from scipy.sparse.linalg import spsolve_triangular as sp_spsolve_triangular
from scipy.sparse import csr_matrix as sp_csr_matrix

from .base import MatrixLinearOperator, _CustomLinearOperator 
from .util import issparse, tosparse
from .linalg import banded_cholesky
from .inv import BandedCholeskyInvOperator

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cp_spsolve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import SuperLU as cp_SuperLU



def banded_cholesky_factor_operator(A):
    """Given an input SPD LinearOperator A, represents the lower-triangular square factor L in the factorization
    A = L L^T computed by a banded Cholesky factorization.
    """

    # Produce a cholesky inverse operator
    Ainv = BandedCholeskyInvOperator(A)

    # Extract its C
    chol_fac = Ainv.chol_fac

    return MatrixLinearOperator(chol_fac)



class BandedCholeskyFactorInvOperator(_CustomLinearOperator):
    """Given an input SPD LinearOperator A, represents L^{-1} where L is the lower-triangular square factor L in the factorization
    A = L L^T computed by a banded Cholesky factorization.

    TO-DO 12/3/24: make this work on GPU.
    """

    def __init__(self, A, _superlu=None):

        assert isinstance(A, MatrixLinearOperator), "Must give MatrixOperator as an input."
        
        # Device
        device = A.device

        # Get the lower triangular factor L
        L = banded_cholesky_factor_operator(A)

        # Make a CSR matrix
        self.Lmat = sp_csr_matrix(L.A)
        self.Lmat_t = sp_csr_matrix(self.Lmat.T)
        n = A.shape[0]
        
        # Build matvec and rmatvec
        if device == "cpu":

            def _matvec(x):
                tmp = sp_spsolve_triangular(self.Lmat, x, lower=True)
                return tmp
            
            def _rmatvec(x):
                tmp = sp_spsolve_triangular(self.Lmat_t, x, lower=False)
                return tmp
            
        else:

            def _matvec(x):
                tmp = cp_spsolve_triangular(self.Lmat, x, lower=True)
                return tmp
            
            def _rmatvec(x):
                tmp = cp_spsolve_triangular(self.Lmat_t, x, lower=False)
                return tmp

        
        super().__init__( (n,n), _matvec, _rmatvec, device=device, dtype=A.dtype)
        
        
    def to_gpu(self):
        
        raise NotImplementedError
    
    
    def to_cpu(self):
        
        raise NotImplementedError







