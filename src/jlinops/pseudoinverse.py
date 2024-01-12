import numpy as np
import math

from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import qr as sp_qr
from scipy.linalg import solve_triangular as sp_solve_triangular
from scipy.sparse.linalg import SuperLU as sp_SuperLU
from scipy.sparse.linalg import cg as sp_cg
import scipy.sparse as sps



from .base import MatrixLinearOperator, _CustomLinearOperator 
from .linalg import banded_cholesky
from .diagonal import DiagonalOperator
from .derivatives import Neumann2D
from .linalg import dct_sqrt_pinv, dct_pinv
from .linear_solvers import cg
from .util import issparse, tosparse, get_device


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

    
    

    
    
    
class CGWeightedNeumann2DPinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, warmstart_prev=True, check=False, which="scipy", *args, **kwargs):

        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        device = get_device(weights)

        # Build R_w
        self.R = Neumann2D(grid_shape, device=device)
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.Rpinv = dct_sqrt_pinv(self.R.T @ self.R, grid_shape)

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = MatrixLinearOperator(W)
        self.Wpinv = QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvOperator(self.Rw, self.W, self.Wpinv, self.Rpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec(x)

        def _rmatvec(x):
            return self.Rwpinv.rmatvec(x)

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def to_gpu(self):
        return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)

    
    
    
    
    
    
    