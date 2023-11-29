import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

from .linear_solvers import cg as jlinops_cg
from .base import MatrixLinearOperator, _CustomLinearOperator


from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as cupy_cg

    

class CGInverseOperator(_CustomLinearOperator):
    """Represents an approximation to the inverse operator of an input SPD or SSPD (with care) LinearOperator,
    """

    def __init__(self, operator, warmstart_prev=True, which="jlinops", check=False, *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Store operator
        self.original_op = operator
        
        # Device
        device = operator.device

        # Setup
        self.which = which
        self.prev_eval = None
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.args = args
        self.kwargs = kwargs
        
        if device == "cpu":
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    return sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = scipy_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    return sol
                
            else:
                raise NotImplementedError
                
        else:
            
            
            if self.which == "jlinops":
                
                def _matvec(x):
                    solver_data = jlinops_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs)
                    sol = solver_data["x"]
                    if self.check:
                        assert solver_data["converged"], "CG algorithm did not converge"
                    return sol
        
            elif self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.original_op, x, x0=self.prev_eval, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    return sol
                
            else:
                raise NotImplementedError
            
            
        super().__init__( self.original_op.shape, _matvec, _matvec, device=device, dtype=self.original_op.dtype)
        
        
        
    def to_gpu(self):
        return CGInverseOperator(operator.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGInverseOperator(operator.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)








