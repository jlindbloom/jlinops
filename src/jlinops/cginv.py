import numpy as np
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator

from scipy.sparse.linalg import cg as scipy_cg

from .linear_solvers import cg



class CGInverseOperator(_CustomLinearOperator):
    """Represents an approximation to the inverse operator of an input SPD LinearOperator,
    """

    def __init__(self, operator, warmstart_prev=True, which="jlinops", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = operator

        # Setup
        self.which = which
        self.prev_eval = None
        self.warmstart_prev = warmstart_prev
        
        # Define matvec and rmatvec
        def _matvec(x):
            if self.which == "scipy":
                sol, converged = scipy_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs) 
                assert converged == 0, "CG algorithm did not converge!"
            elif self.which == "jlinops":
                solver_data = cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs)
                sol = solver_data["x"]
            else:
                raise ValueError

            if self.warmstart_prev:
                self.prev_eval = sol.copy()
            return sol
        
        super().__init__( self.original_op.shape, _matvec, _matvec )








