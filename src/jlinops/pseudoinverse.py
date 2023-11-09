import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator
import scipy.sparse as sps

from .matrix import MatrixOperator
from .sparsematrix import SparseMatrixOperator
from .cholesky import banded_cholesky_factorization

from scipy.sparse.linalg import cg as scipy_cg
from .linear_solvers import cg


class SparseCholeskyApproximatePseudoInverseOperator(_CustomLinearOperator):
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






class CGApproximatePseudoinverse(_CustomLinearOperator):
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
