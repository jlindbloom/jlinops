import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator
import scipy.sparse as sps
import math

from .matrix import MatrixOperator, SparseMatrixOperator
from .util import banded_cholesky_factorization
from .diagonal import DiagonalOperator
from .derivatives import DiscreteGradientNeumann2D
from .dct import build_dct_Lpinv

from scipy.sparse.linalg import cg as scipy_cg
from .linear_solvers import cg


class BandedCholeskyPseudoinverseOperator(_CustomLinearOperator):
    """Takes a (non-square) SciPy sparse matrix and builds a linear operator representing an approximation to the 
    pseudo-inverse of A. This is most efficient if A^T A is sparse and banded.
    """

    def __init__(self, mat_operator, delta=1e-3):

        assert isinstance(mat_operator, SparseMatrixOperator), "must give SparseMatrixOperator as an input."
        
        # Keep original operator
        self.original_op = mat_operator

        # Store delta and shape
        self.delta = delta
        self._k, self._n = self.original_op.A.shape
        self._n = self.original_op.A.shape[1]

        # Perform factorization
        self.chol_fac, self.superlu = banded_cholesky_factorization( (self.original_op.A.T @ self.original_op.A) + self.delta*sps.eye(self._n) )

        # Build matvec and rmatvec
        def _matvec(x):
            tmp = self.original_op.A.T @ x
            tmp = self.superlu.solve(tmp, trans="N")
            return tmp
        
        def _rmatvec(x):
            tmp = self.superlu.solve(x, trans="T")
            tmp = self.original_op.A @ tmp
            return tmp
        
        super().__init__( (self._n, self._k), _matvec, _rmatvec )



class QRPseudoInverseOperator(_CustomLinearOperator):
    """Takes a dense matrix A with full column rank, builds a linear operator representing the pseudo-inverse of A
    using the QR method.
    """

    def __init__(self, mat_operator):

        assert isinstance(mat_operator, MatrixOperator), "must give MatrixOperator as an input."

        # Store original operator
        self.original_op = mat_operator
        self._k, self._n = self.original_op.shape

        # Perform QR decomposition
        self.Q_fac, self.R_fac = scipy_qr(self.original_op.A, mode="economic")

        # Build matvec and rmatvec
        def _matvec(vec):
            tmp = self.Q_fac.T @ vec
            tmp = scipy_solve_triangular(self.R_fac, tmp, lower=False)
            return tmp
        
        def _rmatvec(vec):
            tmp = scipy_solve_triangular(self.R_fac.T, vec, lower=True)
            tmp = self.Q_fac @ tmp
            return tmp

        super().__init__( (self._n, self._k), _matvec, _rmatvec )




class CGPseudoinverseOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method.
    """

    def __init__(self, operator, warmstart_prev=False, which="jlinops", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = operator

        # Setup
        self.which = which
        self.in_shape = self.original_op.shape[0]
        self.out_shape = self.original_op.shape[1]
        self.prev_eval = np.zeros(self.out_shape)
        self.prev_eval_t = np.zeros(self.in_shape)
        self.warmstart_prev = warmstart_prev

        # Build both operators we need
        self.AtA = self.original_op.T @ self.original_op
        self.AAt = self.original_op @ self.original_op.T

        # Define matvec and rmatvec
        def _matvec(x):
            if self.which == "scipy":
                sol, converged = scipy_cg(self.AtA, self.original_op.T @ x, x0=self.prev_eval, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.AtA, self.original_op.T @ x, x0=self.prev_eval, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval = sol.copy()

            return sol
        
        def _rmatvec(x):
            if self.which == "scipy":
                sol, converged = scipy_cg(self.AAt, self.original_op @ x, x0=self.prev_eval_t, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.AAt, self.original_op @ x, x0=self.prev_eval_t, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval_t = sol.copy()
                
            return sol
        
        super().__init__( (self.out_shape, self.in_shape), _matvec, _rmatvec )



class CGModPseudoinverseOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    """

    def __init__(self, operator, W, Wpinv, warmstart_prev=False, which="jlinops", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = operator
        self.W = W
        self.Wpinv = Wpinv

        # Setup
        self.which = which
        self.in_shape = self.original_op.shape[0]
        self.out_shape = self.original_op.shape[1]
        self.prev_eval = np.zeros(self.out_shape)
        self.prev_eval_t = np.zeros(self.out_shape)
        self.warmstart_prev = warmstart_prev

        # Build both operators we need
        self.AtA = self.original_op.T @ self.original_op
        self.AAt = self.original_op @ self.original_op.T

        # Define matvec and rmatvec
        def _matvec(x):
            if self.which == "scipy":
                sol, converged = scipy_cg(self.AtA, self.original_op.T @ x, x0=self.prev_eval, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.AtA, self.original_op.T @ x, x0=self.prev_eval, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval = sol.copy()

            return sol
        
        def _rmatvec(x):

            # Project x onto range(A^T A) = range(A^T).
            x = x - (self.W @ (self.Wpinv @ x))

            if self.which == "scipy":
                sol, converged = scipy_cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.AtA, x, x0=self.prev_eval_t, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval_t = sol.copy()
                
            return self.original_op @ sol
        
        super().__init__( (self.out_shape, self.in_shape), _matvec, _rmatvec )




class CGPreconditionedPseudoinverseOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    """

    def __init__(self, operator, W, Wpinv, Lpinv, warmstart_prev=True, which="jlinops", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = operator
        self.W = W
        self.Wpinv = Wpinv
        self.Lpinv = Lpinv
        self.Ltpinv = Lpinv.T

        # Setup
        self.which = which
        self.in_shape = self.original_op.shape[0]
        self.out_shape = self.original_op.shape[1]
        self.prev_eval = np.zeros(self.out_shape)
        self.prev_eval_t = np.zeros(self.out_shape)
        self.warmstart_prev = warmstart_prev

        # Build both operators we need
        self.AtA = self.original_op.T @ self.original_op
        self.Q = self.Lpinv @ self.AtA @ self.Ltpinv

        # Define matvec and rmatvec
        def _matvec(x):
            if self.which == "scipy":
                sol, converged = scipy_cg(self.Q, self.Lpinv @ (self.original_op.T @ x), x0=self.prev_eval, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.Q, self.Lpinv @ (self.original_op.T @ x), x0=self.prev_eval, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval = sol.copy()

            return self.Ltpinv @ sol
        
        def _rmatvec(x):

            # Project x onto range(A^T A) = range(A^T).
            x = x - (self.W @ (self.Wpinv @ x))

            if self.which == "scipy":
                sol, converged = scipy_cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.Q, self.Lpinv @ x, x0=self.prev_eval_t, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval_t = sol.copy()
                
            return self.original_op @ (self.Ltpinv @ sol)
        
        super().__init__( (self.out_shape, self.in_shape), _matvec, _rmatvec, dtype=np.float64 )




class CGWeightedDiscreteGradientNeumann2DPseudoinverse(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a DiscreteGradientNeumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, warmstart_prev=True, which="jlinops", *args, **kwargs):

        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_size = grid_shape

        # Build R_w
        self.R = DiscreteGradientNeumann2D(grid_shape)
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.Rpinv = build_dct_Lpinv(self.R.T @ self.R, grid_shape)

        # Take care of W (columns span the kernel of R)
        W = np.ones((self.R.shape[1],1))
        self.W = MatrixOperator(W)
        self.Wpinv = QRPseudoInverseOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPseudoinverseOperator(self.Rw, self.W, self.Wpinv, self.Rpinv, warmstart_prev=warmstart_prev, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv @ x

        def _rmatvec(x):
            return self.Rwpinv.T @ x

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64)

