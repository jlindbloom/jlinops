import numpy as np
import math

from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import qr as sp_qr
from scipy.linalg import solve_triangular as sp_solve_triangular
from scipy.sparse.linalg import SuperLU as sp_SuperLU
from scipy.sparse.linalg import cg as sp_cg
import scipy.sparse as sps



from .base import MatrixLinearOperator, _CustomLinearOperator, IdentityOperator
from .linalg.factorizations import banded_cholesky
from .diagonal import DiagonalOperator
from .derivatives import Neumann2D, Dirichlet2DSym
# from .cginv import CGDirichlet2DSymLapInvOperator, CGWeightedDirichlet2DSymLapInvOperator
from .linalg.dct import dct_sqrt_pinv, dct_pinv
from .linalg.dst import dst_sqrt_pinv, dst_pinv
from .linear_solvers import cg
from .util import issparse, tosparse, get_device
from .linalg import dct_sqrt, dct_sqrt_pinv


from .derivatives import get_neumann2d_laplacian_diagonal, get_neumann2d_laplacian_tridiagonal
from .inv import TridiagInvOperator



from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import SuperLU as cp_SuperLU




class BandedCholeskyPinvOperator(_CustomLinearOperator):
    """Takes a (non-square) MatrixLinearOperator A and builds a linear operator representing an approximation to the 
    pseudo-inverse of A. This is most efficient if A^T A is sparse and (already) banded.
    """

    def __init__(self, A, delta=1e-3, _superlu=None):

        assert isinstance(A, MatrixLinearOperator), "Must give MatrixOperator as an input."
        
        # Bind
        self.original_op = A
        self.original_shape = A.shape
        self.delta = delta
        
        # Device
        device = A.device
        
        # Original shape
        k, n = A.shape
        
        # Enforce that underling A.T A is a sparse type
        if not issparse(A.A):
            AtA = MatrixLinearOperator( tosparse(A.A.T @ A.A).tocsc() )
        else:
            AtA = MatrixLinearOperator(A.A.T @ A.A)
        
        # Even if on GPU, factorize and make superlu object on CPU
        if device == "cpu":
            AtA_cpu = AtA
        else:
            AtA_cpu = AtA.to_cpu()
            
        # Matrix we will factorize
        mat = AtA_cpu.A + self.delta*sps.eye(n)
        
        if _superlu is None:
            
            # Perform factorization
            chol_fac, superlu = banded_cholesky( mat )
            
            # Make GPU superlu object if applicable
            if device == "gpu":
                superlu = cp_SuperLU(superlu)
                
        else: 
            superlu = _superlu
            pass
        
        
            
        # Bind superlu and A
        self.superlu = superlu
        self.A = A
            
        # Build matvec and rmatvec
        def _matvec(x):
            tmp = self.A.rmatvec(x)
            tmp = self.superlu.solve(tmp, trans="N")
            return tmp
        
        def _rmatvec(x):
            tmp = self.superlu.solve(x, trans="T")
            tmp = self.A.matvec(tmp)
            return tmp
        
        super().__init__( (n,k), _matvec, _rmatvec, device=device, dtype=A.dtype)
        
        
    def to_gpu(self):
        
        # Switch to CPU superlu
        superlu = cp_SuperLU(self.superlu)
        
        return BandedCholeskyPinvOperator(self._matvecA.to_gpu(), delta=self.delta, _superlu=superlu)
    
    
    def to_cpu(self):
        
        raise NotImplementedError



class QRPinvOperator(_CustomLinearOperator):
    """Takes a dense matrix A with full column rank, builds a linear operator representing the pseudo-inverse of A
    using the QR method.
    """

    def __init__(self, A):

        #assert isinstance(A, MatrixLinearOperator), "must give MatrixOperator as an input."

        if not isinstance(A, MatrixLinearOperator):
            A = MatrixLinearOperator(A)
        
        # Store original operator
        self.original_op = A
        k, n = A.shape
        
        # Device
        device = A.device
        
        if device == "cpu":
            
            Q_fac, R_fac = sp_qr(A.A, mode="economic")

            # Build matvec and rmatvec
            def _matvec(vec):
                tmp = Q_fac.T @ vec
                tmp = sp_solve_triangular(R_fac, tmp, lower=False)
                return tmp

            def _rmatvec(vec):
                tmp = scipy_solve_triangular(R_fac.T, vec, lower=True)
                tmp = Q_fac @ tmp
                return tmp
            
        else:
            
            Q_fac, R_fac = cp_qr(A.A, mode="reduced")

            # Build matvec and rmatvec
            def _matvec(vec):
                tmp = Q_fac.T @ vec
                tmp = cp_solve_triangular(R_fac, tmp, lower=False)
                return tmp

            def _rmatvec(vec):
                tmp = cp_solve_triangular(R_fac.T, vec, lower=True)
                tmp = Q_fac @ tmp
                return tmp

        super().__init__( (n, k), _matvec, _rmatvec , device=device)
        
        
    def to_gpu(self):
        return QRPinvOperator(self.original_op.to_gpu())
    
    def to_cpu(self):
        return QRPinvOperator(self.original_op.to_cpu())




class CGPreconditionedPinvModOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    
    W: a LinearOperator representing a matrix with linearly independent columns that spans null(A).
    Wpinv: a LinearOperator represening the pseudoinverse of W.
    Lpinv: 
    """

    def __init__(self, A, W, Wpinv, Mpinv, warmstart_prev=False, check=False, which="scipy", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Device
        device = A.device
        
        # Store operator
        self.A = A
        self.W = W
        self.Wpinv = Wpinv
        self.Mpinv = Mpinv
        
        # Shape
        m, n = A.shape
        shape = (n, m)

        # Setup
        self.which = which
        self.check = check
        self.warmstart_prev = warmstart_prev
        self.in_shape = self.A.shape[0]
        self.out_shape = self.A.shape[1]
        
        if device == "cpu":
            self.prev_eval = np.zeros(self.out_shape)
            self.prev_eval_t = np.zeros(self.out_shape)
        else:
            self.prev_eval = cp.zeros(self.out_shape)
            self.prev_eval_t = cp.zeros(self.out_shape)

        # Build both operators we need
        self.C = self.A.T @ self.A
        self.MpinvC = self.Mpinv @ self.C

        
        if device == "cpu":
            
           
            if self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = sp_cg(self.C, self.A.rmatvec(x), x0=self.prev_eval, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    if (self.W is not None) and (self.Wpinv is not None):
                        z = x - (self.W @ (self.Wpinv @ x))
                    else:
                        z = x 
                    
                    sol, converged = sp_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ sol
                
            else:
                raise NotImplementedError
                
        else:
            
            if self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.C, self.A.rmatvec(x), M=self.Mpinv, x0=self.prev_eval, *args, **kwargs)
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):

                    # Project x onto range(A^T A) = range(A^T).
                    if (self.W is not None) and (self.Wpinv is not None):
                        z = x - (self.W @ (self.Wpinv @ x))
                    else:
                        z = x
                    
                    sol, converged = cupy_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ sol
                
            else:
                raise NotImplementedError
        
        
        super().__init__( shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        
        
    def to_gpu(self):
        return CGPreconditionedPinvModOperator(self.A.to_gpu(), self.W.to_gpu(), self.Wpinv.to_gpu(), self.Mpinv.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
        
    def to_cpu(self):
        return CGPreconditionedPinvModOperator(self.A.to_cpu(), self.W.to_cpu(), self.Wpinv.to_cpu(), self.Mpinv.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    


class CGWeightedNeumann2DPinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dct_eps = dct_eps
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        device = get_device(weights)

        # Build R and R_w
        self.R = Neumann2D(grid_shape, device=device)
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct_pinv( self.RtR, grid_shape, eps=dct_eps )

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = MatrixLinearOperator(W)
        self.Wpinv = QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec(x)

        def _rmatvec(x):
            return self.Rwpinv.rmatvec(x)

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def to_gpu(self):
        return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    



class CGWeightedNeumann2DSimplePinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, which="diagonal", *args, **kwargs):

        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        assert which in ["diagonal", "tridiagonal"], f"which must be one of diagonal, tridiagonal."
        self.weights = weights
        self.grid_shape = grid_shape
        self.n = math.prod(grid_shape)
        # self.warmstart_prev = warmstart_prev
        # self.check = check
        self.which = which
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        self.device = get_device(weights)

        # Build R and R_w
        self.R = Neumann2D(grid_shape, device=self.device)
        self.m = self.R.shape[0]
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R
        self.C = self.Rw.T @ self.Rw
        

        # Set up preconditioning operator
        if self.which == "diagonal":

            # Get diagonal
            d = get_neumann2d_laplacian_diagonal(self.grid_shape)
            M = DiagonalOperator(1.0/d)

        elif self.which == "tridiagonal":
            
            l, d, u = get_neumann2d_laplacian_tridiagonal(self.grid_shape)
            M = TridiagInvOperator(l, d, u)

        else:

            raise NotImplementedError

        self.M = M

        def _matvec(x):

            tmp = self.Rw.rmatvec(x)
            sol, converged = sp_cg( self.C, tmp, x0=None, M=self.M, *args, **kwargs) 
            
            return sol
        
        def _rmatvec(x):

            sol, converged = sp_cg( self.C, x, x0=None, M=M, *args, **kwargs) 
            tmp = self.Rw.matvec(sol)

            return tmp

        
        super().__init__( (self.n, self.m), _matvec, _rmatvec, dtype=np.float64, device="cpu")

    def to_gpu(self):
        return CGWeightedNeumann2DSimplePinvOperator( self.grid_shape, cp.asarray(self.weights), which=self.which, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGWeightedNeumann2DSimplePinvOperator( self.grid_shape, cp.asnumpy(self.weights), which=self.which, *self.args, **self.kwargs)
    











class CGWeightedDirichlet2DSymPinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Dirichlet2DSym operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, warmstart_prev=False, check=False, which="scipy", dst_eps=1e-14, *args, **kwargs):

        device = get_device(weights)
        self.R = Dirichlet2DSym(grid_shape, device=device)
        assert self.R.shape[0] == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dst_eps = dst_eps
        self.args = args
        self.kwargs = kwargs
        
       
        # Build RtR and Rw
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dst_pinv( self.RtR, grid_shape, eps=self.dst_eps )

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = MatrixLinearOperator(W)
        self.Wpinv = QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, dst_eps=self.dst_eps, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec(x)

        def _rmatvec(x):
            return self.Rwpinv.rmatvec(x)

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def to_gpu(self):
        return CGWeightedDirichlet2DSymPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    

    def to_cpu(self):
        return CGWeightedDirichlet2DSymPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    











###################################################################################################################
### These methods use the singular preconditioner based on the Neumann problem to solve the Dirichlet problem. ####
###################################################################################################################
    


class CGDirichlet2DSymSconLapInvOperator(_CustomLinearOperator):
    """Represents the inverse of the Laplacian operator R^T R
    equipped with Dirichlet boundary conditions.
    """

    def __init__(self, grid_shape, check=True, *args, **kwargs):

        # Make both operators
        self.grid_shape = grid_shape
        self.n = math.prod(grid_shape)
        self.Rd = Dirichlet2DSym(grid_shape)
        self.Rn = Neumann2D(grid_shape) 

        # Misc
        self.check = check

        # Build Dirichlet Laplacian
        self.A = self.Rd.T @ self.Rd
        self.M = self.Rn.T @ self.Rn

        # Matrix spanning the kernel of the Neumann operator
        self.Wmat = np.ones((self.n,1))
        self.W = MatrixLinearOperator(self.Wmat)

        # Operator for getting the part of solution in the kernel
        #self.sol_ker_op = self.Wmat @ np.linalg.inv( self.Wmat.T @ ( self.A @ self.Wmat )  ) @ self.Wmat.T
        self.sol_ker_op = self.W @ ( MatrixLinearOperator( np.linalg.inv( self.Wmat.T @ (self.A @ self.Wmat)  ) ) @ self.W.T )

        # Get L, Lpseudo, and Loblique
        self.L = dct_sqrt(self.M, self.grid_shape).T
        self.Lpinv = dct_sqrt_pinv(self.M, self.grid_shape).T
        self.RdWpinv = QRPinvOperator(self.Rd @ self.Wmat)
        self.Loblique = ( IdentityOperator( (self.n,self.n) ) - (self.W @ self.RdWpinv @ self.Rd) ) @ self.Lpinv
 
        # Set of coefficient matrix
        self.C = self.Loblique.T @ self.A @ self.Loblique


        def _matvec(x):

            sol, converged = sp_cg( self.C, self.Loblique.T @ x, x0=None, M=None, *args, **kwargs) 
            if self.check:
                assert converged == 0, "CG algorithm did not converge!"
            
            # Project onto range of L
            sol = self.L @ self.Lpinv @ sol
            
            part_one = self.Loblique @ sol
            part_two = self.sol_ker_op @ x

            result = part_one + part_two

            return result

        super().__init__( (self.n, self.n), _matvec, _matvec, dtype=np.float64, device="cpu")




class CGWeightedDirichlet2DSymSconLapInvOperator(_CustomLinearOperator):
    """Represents the inverse of the weighted Laplacian operator R^T D_w R
    equipped with Dirichlet boundary conditions.
    """

    def __init__(self, grid_shape, weights, check=True, *args, **kwargs):

        # Make both operators
        self.grid_shape = grid_shape
        self.Rn = Neumann2D(grid_shape)
        self.n = self.Rn.shape[1] 
        self.Rd = DiagonalOperator(np.sqrt(weights)) @ Dirichlet2DSym(grid_shape)
        self.weights = weights

        # Misc
        self.check = check

        # Build Dirichlet Laplacian
        self.A = self.Rd.T @ self.Rd
        self.M = self.Rn.T @ self.Rn

        # Matrix spanning the kernel of the Neumann operator
        self.Wmat = np.ones((self.n, 1))
        self.W = MatrixLinearOperator(self.Wmat)

        # Operator for getting the part of solution in the kernel
        #self.sol_ker_op = self.Wmat @ np.linalg.inv( self.Wmat.T @ ( self.A @ self.Wmat )  ) @ self.Wmat.T
        self.sol_ker_op = self.W @ ( MatrixLinearOperator( np.linalg.inv( self.Wmat.T @ (self.A @ self.Wmat)  ) ) @ self.W.T )

        # Get L, Lpseudo, and Loblique
        self.L = dct_sqrt(self.M, self.grid_shape).T
        self.Lpinv = dct_sqrt_pinv(self.M, self.grid_shape).T
        self.RdWpinv = QRPinvOperator(self.Rd @ self.Wmat)
        self.Loblique = ( IdentityOperator( (self.n,self.n) ) - (self.W @ self.RdWpinv @ self.Rd) ) @ self.Lpinv
 
        # Set of coefficient matrix
        self.C = self.Loblique.T @ self.A @ self.Loblique


        def _matvec(x):

            sol, converged = sp_cg( self.C, self.Loblique.T @ x, x0=None, M=None, *args, **kwargs) 
            if self.check:
                assert converged == 0, "CG algorithm did not converge!"
            
            # Project onto range of L
            sol = self.L @ self.Lpinv @ sol
            
            part_one = self.Loblique @ sol
            part_two = self.sol_ker_op @ x

            result = part_one + part_two

            return result


        super().__init__( (self.n, self.n), _matvec, _matvec, dtype=np.float64, device="cpu")






###############################
### For Dirichlet gradients ###
###############################

class CGDirichlet2DSymSconPinvOperator(_CustomLinearOperator):

    def __init__(self, grid_shape, check=True, *args, **kwargs):

        # Make operator
        self.R = Dirichlet2DSym(grid_shape)
        m, n = self.R.shape

        # Make Laplacian solver
        self.inv_lap = CGDirichlet2DSymSconLapInvOperator(grid_shape=grid_shape, check=check, *args, **kwargs)

        def _matvec(x):

            tmp = self.R.rmatvec(x)
            result = self.inv_lap.matvec(tmp)

            return result
        

        def _rmatvec(x):

            tmp = self.inv_lap.matvec(x)
            result = self.R.matvec(tmp)

            return result
        

        super().__init__( (n, m), _matvec, _rmatvec, dtype=np.float64, device="cpu")




class CGWeightedDirichlet2DSymSconPinvOperator(_CustomLinearOperator):

    def __init__(self, grid_shape, weights, check=True, *args, **kwargs):

        # Make operator
        self.R = DiagonalOperator(np.sqrt(weights)) @ Dirichlet2DSym(grid_shape)
        self.weights = weights
        m, n = self.R.shape

        # Make Laplacian solver
        self.inv_lap = CGWeightedDirichlet2DSymSconLapInvOperator(grid_shape=grid_shape, weights=weights, check=check, *args, **kwargs)

        def _matvec(x):

            tmp = self.R.rmatvec(x)
            result = self.inv_lap.matvec(tmp)

            return result
        

        def _rmatvec(x):

            tmp = self.inv_lap.matvec(x)
            result = self.R.matvec(tmp)

            return result
        

        super().__init__( (n, m), _matvec, _rmatvec, dtype=np.float64, device="cpu")











##################################
### NEED TO CHECK ALL OF THESE ###
##################################

class CGPinvOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method.
    """

    def __init__(self, A, warmstart_prev=False, which="scipy", check=False, *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = A
        self.A = A
        
        # Device
        device = A.device
        
        # Shape
        m, n = A.shape
        shape = (n, m)
    
        # Setup
        self.which = which
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.args = args
        self.kwargs = kwargs
        self.in_shape = A.shape[0]
        self.out_shape = A.shape[1]
        
        if device == "cpu":
            self.prev_eval = np.zeros(self.out_shape)
            self.prev_eval_t = np.zeros(self.in_shape)
        else:
            self.prev_eval = cp.zeros(self.out_shape)
            self.prev_eval_t = cp.zeros(self.in_shape)
            
        self.warmstart_prev = warmstart_prev

        # Build both operators we need
        self.AtA = self.original_op.T @ self.original_op
        self.AAt = self.original_op @ self.original_op.T
        
        
        if device == "cpu":
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    solver_data = jlinops_cg(self.AAt, self.A.matvec(x), x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = sp_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                        
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    sol, converged = sp_cg(self.AAt, self.A.matvec(x), x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return sol
                
            else:
                raise NotImplementedError
                
        else:
            
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):
                    solver_data = jlinops_cg(self.AAt, self.A.matvec(x), x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):
                    sol, converged = cupy_cg(self.AAt, self.A.matvec(x), x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return sol
                
            else:
                raise NotImplementedError
            
            
        super().__init__( shape, _matvec, _rmatvec, device=device, dtype=self.A.dtype)
        
        
    def to_gpu(self):
        return CGPinvOperator(self.A.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGPinvOperator(self.A.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)


    
class CGModPinvOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    
    W: a LinearOperator representing a matrix with linearly independent columns that spans null(A).
    Wpinv: a LinearOperator represening the pseudoinverse of W.
    """

    def __init__(self, A, W, Wpinv, warmstart_prev=False, which="scipy", check=False, *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = A
        self.A = A
        self.W = W
        self.Wpinv = Wpinv
        
        # Device
        device = A.device
    
        # Shape
        m, n = A.shape
        shape = (n, m)
    
        # Setup
        self.which = which
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.args = args
        self.kwargs = kwargs
        self.in_shape = A.shape[0]
        self.out_shape = A.shape[1]
        
        if device == "cpu":
            self.prev_eval = np.zeros(self.out_shape)
            self.prev_eval_t = np.zeros(self.in_shape)
        else:
            self.prev_eval = cp.zeros(self.out_shape)
            self.prev_eval_t = cp.zeros(self.in_shape)
            
        self.warmstart_prev = warmstart_prev

        # Build both operators we need
        self.AtA = self.original_op.T @ self.original_op
        self.AAt = self.original_op @ self.original_op.T
        
        
        if device == "cpu":
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    solver_data = jlinops_cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = sp_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                        
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    sol, converged = sp_cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ sol
                
            else:
                raise NotImplementedError
                
        else:
            
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    solver_data = jlinops_cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.AtA, self.A.rmatvec(x), x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    sol, converged = cupy_cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ sol
                
            else:
                raise NotImplementedError
            
            
        super().__init__( shape, _matvec, _rmatvec, device=device, dtype=self.A.dtype)
        
        
        
    def to_gpu(self):
        return CGModPinvOperator(self.A.to_gpu(), self.W.to_gpu(), self.Wpinv.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGPModinvOperator(self.A.to_cpu(), self.W.to_cpu(), self.Wpinv.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)

    
    
class CGPreconditionedPinvOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    
    W: a LinearOperator representing a matrix with linearly independent columns that spans null(A).
    Wpinv: a LinearOperator represening the pseudoinverse of W.
    Lpinv: 
    """

    def __init__(self, A, W, Wpinv, Lpinv, warmstart_prev=True, check=False, which="scipy", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Device
        device = A.device
        
        # Store operator
        self.A = A
        self.W = W
        self.Wpinv = Wpinv
        self.Lpinv = Lpinv
        self.Ltpinv = Lpinv.T
        
        # Shape
        m, n = A.shape
        shape = (n, m)

        # Setup
        self.which = which
        self.check = check
        self.warmstart_prev = warmstart_prev
        self.in_shape = self.A.shape[0]
        self.out_shape = self.A.shape[1]
        
        if device == "cpu":
            self.prev_eval = np.zeros(self.out_shape)
            self.prev_eval_t = np.zeros(self.out_shape)
        else:
            self.prev_eval = cp.zeros(self.out_shape)
            self.prev_eval_t = cp.zeros(self.out_shape)

        # Build both operators we need
        self.AtA = self.A.T @ self.A
        self.Q = self.Lpinv @ self.AtA @ self.Ltpinv

        
        if device == "cpu":
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.Q, self.Lpinv @ (self.A.rmatvec(x)), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return self.Ltpinv @ sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    solver_data = jlinops_cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ (self.Ltpinv @ sol)
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = sp_cg(self.Q, self.Lpinv @ (self.A.rmatvec(x)), x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return self.Ltpinv @ sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    sol, converged = sp_cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ (self.Ltpinv @ sol)
                
            else:
                raise NotImplementedError
                
        else:
            
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.Q, self.Lpinv @ (self.A.rmatvec(x)), x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                        
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return self.Ltpinv @ sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    solver_data = jlinops_cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ (self.Ltpinv @ sol)
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.Q, self.Lpinv @ (self.A.rmatvec(x)), x0=self.prev_eval, *args, **kwargs)
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return self.Ltpinv @ sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    x = x - (W @ (Wpinv @ x))
                    
                    sol, converged = cupy_cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ (self.Ltpinv @ sol)
                
            else:
                raise NotImplementedError
        
        
        super().__init__( shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        
        
    def to_gpu(self):
        return CGPreconditionedPinvOperator(self.A.to_gpu(), self.W.to_gpu(), self.Wpinv.to_gpu(), self.Lpinv.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
        
    def to_cpu(self):
        return CGPreconditionedPinvOperator(self.A.to_cpu(), self.W.to_cpu(), self.Wpinv.to_cpu(), self.Lpinv.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    

    
    
class Neumann2DPinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse R^\dagger where R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied using a DCT transform method. 
    """
    
    def __init__(self, grid_shape, device="cpu", eps=1e-14):
        
        # Grid shape
        self.grid_shape = grid_shape
        self.eps = eps
        
        # Neumann2D operator
        self.R = Neumann2D(grid_shape, device=device)
        self.Apinv = dct_pinv(self.R.T @ self.R, grid_shape, eps=self.eps)
        
        def _matvec(x):
            return self.Apinv.matvec(x)

        def _rmatvec(x):
            return self.Apinv.rmatvec(x)
        
        super().__init__( self.Apinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        
    def to_gpu(self):
        return Neumann2DPinvOperator(self.grid_shape, device="gpu", eps=self.eps)
    
    def to_cpu(self):
        return Neumann2DPinvOperator(self.grid_shape, device="cpu", eps=self.eps)

    
    

    
    
    
# class CGWeightedNeumann2DPinvOperator(_CustomLinearOperator):
#     """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
#     D_w is a diagonal matrix of weights and R is a Neumann2D operator.
#     Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
#     gradient method, where the preconditioner is based on the operator with identity weights. 
#     """

#     def __init__(self, grid_shape, weights, warmstart_prev=True, check=False, which="scipy", *args, **kwargs):

#         assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
#         self.weights = weights
#         self.grid_shape = grid_shape
#         self.warmstart_prev = warmstart_prev
#         self.check = check
#         self.which = which
#         self.args = args
#         self.kwargs = kwargs
        
#         # Figure out device
#         device = get_device(weights)

#         # Build R_w
#         self.R = Neumann2D(grid_shape, device=device)
#         self.Dw = DiagonalOperator(weights)
#         self.Rw = self.Dw @ self.R

#         # Get Rpinv (with identity weights)
#         self.Rpinv = dct_sqrt_pinv(self.R.T @ self.R, grid_shape)

#         # Take care of W (columns span the kernel of R)
#         if device == "cpu":
#             W = np.ones((self.R.shape[1],1))
#         else:
#             W = cp.ones((self.R.shape[1],1))
            
#         self.W = MatrixLinearOperator(W)
#         self.Wpinv = QRPinvOperator(self.W)

#         # Make Rwpinv
#         self.Rwpinv = CGPreconditionedPinvOperator(self.Rw, self.W, self.Wpinv, self.Rpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

#         def _matvec(x):
#             return self.Rwpinv.matvec(x)

#         def _rmatvec(x):
#             return self.Rwpinv.rmatvec(x)

#         super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

#     def to_gpu(self):
#         return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    
#     def to_cpu(self):
#         return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)

    
    
    
    
    
    
    