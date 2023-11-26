import numpy as np

from scipy.sparse.linalg._interface import _CustomLinearOperator



class DiagonalOperator(_CustomLinearOperator):
    """Implements a diagonal linear operator D.
    """

    def __init__(self, diagonal):
        
        self.diagonal = diagonal
        n = len(diagonal)

        def _matvec(x):
            return self.diagonal*x
        
        super().__init__( (n,n), _matvec, _matvec)





