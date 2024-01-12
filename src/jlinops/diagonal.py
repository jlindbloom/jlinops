import numpy as np
# from scipy.sparse.linalg._interface import _CustomLinearOperator

from .base import LinearOperator, _CustomLinearOperator
from .util import get_device

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp



class DiagonalOperator(_CustomLinearOperator):
    """Implements a diagonal linear operator D.
    """

    def __init__(self, diagonal):
        
        assert diagonal.ndim == 1, "Diagonal array must be 1-dimensional."
        self.diagonal = diagonal
        n = len(diagonal)
        device = get_device(diagonal)

        def _matvec(x):
            return self.diagonal*x
        
        def _rmatvec(x):
            return self.diagonal.conj()*x
        
        super().__init__( (n,n), _matvec, _rmatvec, device=device, dtype=self.diagonal.dtype)

    def to_gpu(self):

        assert CUPY_INSTALLED, "CuPy must be installed!"

        return DiagonalOperator(cp.asarray(self.diagonal))
    
    def to_cpu(self):

        assert CUPY_INSTALLED, "CuPy must be installed!"

        return DiagonalOperator(cp.asnumpy(self.diagonal)) 



