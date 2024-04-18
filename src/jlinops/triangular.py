import numpy as np

from scipy.linalg import solve_triangular as sp_solve_triangular

from .base import _CustomLinearOperator, get_device





class UpperTriangularInverse(_CustomLinearOperator):
    """Represents the inverse of an upper triangular dense matrix.

    U: upper triangular dense array U (lower triangular part will be discarded).
    """

    def __init__(self, U):

        device = get_device(U)
        self.U = np.triu(U.A)
        m = U.shape[0]
        n = U.shape[1]
        assert m == n, "U must be square."
        self.n = n

        def _matvec(x):
            return sp_solve_triangular(self.U, x, trans=0)

        def _rmatvec(x):

            return sp_solve_triangular(self.U, x, trans=1)
        
        super().__init__( (self.n, self.n), _matvec, _rmatvec, dtype=None, device=device)

    def to_gpu(self):
        raise NotImplementedError
    



class LowerTriangularInverse(_CustomLinearOperator):
    """Represents the inverse of an lower triangular dense matrix.

    L: upper triangular dense array L (upper triangular part will be discarded).
    """

    def __init__(self, L):

        device = get_device(L)
        self.L = np.tril(L.A)
        m = L.shape[0]
        n = L.shape[1]
        assert m == n, "L must be square."
        self.n = n

        def _matvec(x):
            return sp_solve_triangular(self.L, x, trans=0, lower=True)

        def _rmatvec(x):

            return sp_solve_triangular(self.L, x, trans=1, lower=True)
        
        super().__init__( (self.n, self.n), _matvec, _rmatvec, dtype=None, device=device)

    def to_gpu(self):
        raise NotImplementedError
    












