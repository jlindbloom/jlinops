import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

import math

from .linear_solvers import cg as jlinops_cg
from .base import MatrixLinearOperator, _CustomLinearOperator, get_device
from .diagonal import DiagonalOperator
from .linalg import dct_pinv
from .pseudoinverse import CGPreconditionedPinvModOperator, QRPinvOperator
from .derivatives import Dirichlet2DSym



from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as cupy_cg

    
    

class CGInverseOperator(_CustomLinearOperator):
    """Represents an approximation to the inverse operator of an input SPD or SSPD (with care) LinearOperator,
    """

    def __init__(self, operator, warmstart_prev=True, which="scipy", check=False, *args, **kwargs):

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
        return CGInverseOperator(self.original_op.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGInverseOperator(self.original_op.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)













   

class CGWeightedDirichlet2DSymInvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Dirichlet2DSym operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 
    """

    def __init__(self, grid_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        # Figure out device
        device = get_device(weights)

        # Some setup

        self.R = Dirichlet2DSym(grid_shape, device=device)
        assert self.R.shape[0] == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.args = args
        self.kwargs = kwargs
        

        # Build R and R_w
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct_pinv( self.RtR, grid_shape, eps=dct_eps )

        # # Take care of W (columns span the kernel of R)
        # if device == "cpu":
        #     W = np.ones((self.R.shape[1],1))
        # else:
        #     W = cp.ones((self.R.shape[1],1))
            
        self.W = None
        self.Wpinv = None

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec(x)

        def _rmatvec(x):
            return self.Rwpinv.rmatvec(x)

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def to_gpu(self):
        return CGWeightedDirichlet2DSymInvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    
    def to_cpu(self):
        return CGWeightedDirichlet2DSymInvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, *self.args, **self.kwargs)
    










