import numpy as np

from scipy.linalg import solve_banded as sp_solve_banded
from scipy.sparse.linalg import cg as sp_cg

from .base import MatrixLinearOperator, _CustomLinearOperator
from .util import issparse, tosparse, get_device
from .linalg import banded_cholesky
from .derivatives import Dirichlet2DSym
from .diagonal import DiagonalOperator
from .linalg import dst_pinv


from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    #from cupyx.scipy.linalg import solve_banded as cp_solve_banded
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








class TridiagInvOperator(_CustomLinearOperator):
    """Represents the inverse of a tridiagonal matrix.
    l: lower off diagonal
    d: diagonal
    u: upper off diagonal
    """

    def __init__(self, l, d, u):

        self.n = len(d)
        assert len(l) == self.n - 1, "lower diagonal inconsistent."
        assert len(u) == self.n - 1, "upper diagonal inconsistent."

        # Store diagonals
        self.l = l
        self.d = d
        self.u = u 

        # Figure out device
        self.device = get_device(d)

        if self.device == "cpu":

            # Helper matrix
            self.Z = np.zeros((3, self.n))
            self.Z[0, 1:] = self.u  # Superdiagonal
            self.Z[1, :] = self.d  # Main diagonal
            self.Z[2, :-1] = self.l  # Subdiagonal
            self.Zt = self.Z.T

            def _matvec(x):

                sol = sp_solve_banded((1, 1), self.Z, x)
                
                return sol
            
            def _rmatvec(x):

                sol = sp_solve_banded((1, 1), self.Zt, x)
                
                return sol
            
        else:

            # Helper matrix
            self.Z = cp.zeros((3, self.n))
            self.Z[0, 1:] = self.u  # Superdiagonal
            self.Z[1, :] = self.d  # Main diagonal
            self.Z[2, :-1] = self.l  # Subdiagonal
            self.Zt = self.Z.T

            def _matvec(x):

                sol = cp_solve_banded((1, 1), self.Z, x)
                
                return sol
        
            def _rmatvec(x):

                sol = cp_solve_banded((1, 1), self.Zt, x)
                
                return sol
            
        
        super().__init__( (self.n, self.n), _matvec, _rmatvec, dtype=None, device=self.device)


    def to_gpu(self):
        
        raise NotImplementedError
        l = cp.asarray(self.l)
        d = cp.asarray(self.d)
        u = cp.asarray(self.u)
        return TridiagInvOperator(l, d, u)
    
    def to_cpu(self):
        l = cp.asnumpy(self.l)
        d = cp.asnumpy(self.d)
        u = cp.asnumpy(self.u)
        return TridiagInvOperator(l, d, u)
    




    
class Dirichelt2DSymLaplacianInv(_CustomLinearOperator):
    """Represents the inverse of the Laplacian R^T R, where R is a Dirichlet2D operator.
    Here matvecs/rmatvecs are applied using a DST transform method. 
    """
    
    def __init__(self, grid_shape, device="cpu", eps=1e-14):
        
        # Grid shape
        self.grid_shape = grid_shape
        self.eps = eps
        
        # Neumann2D operator
        self.R = Dirichlet2DSym(grid_shape, device=device)
        self.Apinv = dst_pinv(self.R.T @ self.R, grid_shape, eps=self.eps)
        
        def _matvec(x):
            return self.Apinv.matvec(x)

        def _rmatvec(x):
            return self.Apinv.rmatvec(x)
        
        super().__init__( self.Apinv.shape, _matvec, _matvec, dtype=np.float64, device=device)
        
    def to_gpu(self):
        return Dirichelt2DSymLaplacianInv(self.grid_shape, device="gpu", eps=self.eps)
    
    def to_cpu(self):
        return Dirichelt2DSymLaplacianInv(self.grid_shape, device="cpu", eps=self.eps)







class CGWeightedDirichlet2DSymLaplacianInv(_CustomLinearOperator):
    """Represents the inverse of the weighted Laplacian operator R^T D_w R
    equipped with Dirichlet boundary conditions.
    """

    def __init__(self, grid_shape, weights, check=True, *args, **kwargs):

        # Make both operators
        self.grid_shape = grid_shape
        self.R = Dirichlet2DSym(grid_shape)
        self.n = self.R.shape[1] 
        self.weights = weights
        self.Lw = self.R.T @ (DiagonalOperator(self.weights) @ self.R)
        self.args = args
        self.kwargs = kwargs

        # Misc
        self.check = check

        # Build preconditioner
        self.M = Dirichelt2DSymLaplacianInv(self.grid_shape, device="cpu")

        def _matvec(x):

            sol, converged = sp_cg( self.Lw, x, x0=np.zeros(self.n), M=self.M, *self.args, **self.kwargs) 
            if self.check:
                assert converged == 0, "CG algorithm did not converge!"
           
            return sol


        super().__init__( (self.n, self.n), _matvec, _matvec, dtype=np.float64, device="cpu")





