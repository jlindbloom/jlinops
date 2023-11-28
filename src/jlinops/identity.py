import numpy as np
from scipy.sparse.linalg._interface import _CustomLinearOperator



class IdentityOperator(_CustomLinearOperator):
    """Represents the identity operator.
    """

    def __init__(self, n):

        def _matvec(x):
            return x
        
        def _rmatvec(x):
            return x
        
        super().__init__( (n, n), _matvec, _rmatvec )