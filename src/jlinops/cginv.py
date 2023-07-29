import numpy as np
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator

from scipy.sparse.linalg import cg as scipy_cg



class CGApproximateInverseOperator(_CustomLinearOperator):
    """Represents an approximation to the inverse operator of an input SPD LinearOperator,
    """
    def __init__(self, operator, warmstart_prev=True, *args, **kwargs):

        # Store operator
        self.original_op = operator

        # Setup
        self.prev_eval = None
        self.warmstart_prev = warmstart_prev
        
        # Define matvec and rmatvec
        def _matvec(x):
            sol, converged = scipy_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs) 
            assert converged == 0, "CG algorithm did not converge!"
            if self.warmstart_prev:
                self.prev_eval = sol.copy()
            return sol
        
        super().__init__( self.original_op.shape, _matvec, _matvec )

    def _inv(self):
        return self.original_op
    
    Inv = property(_inv)









