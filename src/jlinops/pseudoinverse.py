import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator
import scipy.sparse as sps

from .matrix import MatrixOperator
from .sparsematrix import SparseMatrixOperator
from .cholesky import banded_cholesky_factorization



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
        self.chol_fac, self.superlu = banded_cholesky_factorization( (self.original_op.A.T @ self.original_op.A) + self.delta*sps.eye(self._n))

        # Build matvec and rmatvec
        def _matvec(x):
            tmp = self.A.T @ x
            tmp = self.superlu.solve(tmp, trans="N")
            return tmp
        
        def _rmatvec(x):
            tmp = self.superlu.solve(x, trans="T")
            tmp = self.original_op.A @ x
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
            tmp = scipy_solve_triangular(self.R_fac, tmp)
            return tmp
        
        def _rmatvec(vec):
            tmp = scipy_solve_triangular(self.R_fac.T, vec, lower=True)
            tmp = self.Q_fac @ tmp
            return tmp

        super().__init__( (self._n, self._k), _matvec, _rmatvec )


