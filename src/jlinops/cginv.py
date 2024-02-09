import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

import math

from .linear_solvers import cg as jlinops_cg
from .base import MatrixLinearOperator, _CustomLinearOperator, IdentityOperator, get_device
from .diagonal import DiagonalOperator
# from .linalg import dct_sqrt, dct_sqrt_pinv
from .pseudoinverse import CGPreconditionedPinvModOperator, QRPinvOperator
from .derivatives import Dirichlet2DSym, Neumann2D


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













# ###################################################################################################################
# ### These methods use the singular preconditioner based on the Neumann problem to solve the Dirichlet problem. ####
# ###################################################################################################################
    

# class CGDirichlet2DSymLapInvOperator(_CustomLinearOperator):
#     """Represents the inverse of the Laplacian operator R^T R
#     equipped with Dirichlet boundary conditions.
#     """

#     def __init__(self, grid_shape, check=True, *args, **kwargs):

#         # Make both operators
#         self.grid_shape = grid_shape
#         self.n = math.prod(grid_shape)
#         self.Rd = Dirichlet2DSym(grid_shape)
#         self.Rn = Neumann2D(grid_shape) 

#         # Misc
#         self.check = check

#         # Build Dirichlet Laplacian
#         self.A = self.Rd.T @ self.Rd
#         self.M = self.Rn.T @ self.Rn

#         # Matrix spanning the kernel of the Neumann operator
#         self.Wmat = np.ones((self.n,1))
#         self.W = MatrixLinearOperator(self.Wmat)

#         # Operator for getting the part of solution in the kernel
#         self.sol_ker_op = self.Wmat @ np.linalg.inv( self.Wmat.T @ ( self.A @ self.Wmat )  ) @ self.Wmat.T

#         # Get L, Lpseudo, and Loblique
#         self.L = dct_sqrt(self.M, self.grid_shape).T
#         self.Lpinv = dct_sqrt_pinv(self.M, self.grid_shape).T
#         self.RdWpinv = QRPinvOperator(self.Rd @ self.Wmat)
#         self.Loblique = ( IdentityOperator( (self.n,self.n) ) - (self.W @ self.RdWpinv @ self.Rd) ) @ self.Lpinv
 
#         # Set of coefficient matrix
#         self.C = self.Loblique.T @ self.A @ self.Loblique


#         def _matvec(x):

#             sol, converged = scipy_cg( self.C, self.Loblique.T @ x, x0=None, M=None, *args, **kwargs) 
#             if self.check:
#                 assert converged == 0, "CG algorithm did not converge!"
            
#             # Project onto range of L
#             sol = self.L @ self.Lpinv @ sol
            
#             part_one = self.Loblique @ sol
#             part_two = self.sol_ker_op @ x

#             result = part_one + part_two

#             return result

#         super().__init__( (self.n, self.n), _matvec, _matvec, dtype=np.float64, device="cpu")




# class CGWeightedDirichlet2DSymLapInvOperator(_CustomLinearOperator):
#     """Represents the inverse of the weighted Laplacian operator R^T D_w R
#     equipped with Dirichlet boundary conditions.
#     """

#     def __init__(self, grid_shape, weights, check=True, *args, **kwargs):

#         # Make both operators
#         self.grid_shape = grid_shape
#         self.Rn = Neumann2D(grid_shape)
#         self.n = self.Rn.shape[1] 
#         self.Rd = DiagonalOperator(np.sqrt(weights)) @ Dirichlet2DSym(grid_shape)
#         self.weights = weights

#         # Misc
#         self.check = check

#         # Build Dirichlet Laplacian
#         self.A = self.Rd.T @ self.Rd
#         self.M = self.Rn.T @ self.Rn

#         # Matrix spanning the kernel of the Neumann operator
#         self.Wmat = np.ones((self.n, 1))
#         self.W = MatrixLinearOperator(self.Wmat)

#         # Operator for getting the part of solution in the kernel
#         self.sol_ker_op = self.Wmat @ np.linalg.inv( self.Wmat.T @ ( self.A @ self.Wmat )  ) @ self.Wmat.T

#         # Get L, Lpseudo, and Loblique
#         self.L = dct_sqrt(self.M, self.grid_shape).T
#         self.Lpinv = dct_sqrt_pinv(self.M, self.grid_shape).T
#         self.RdWpinv = QRPinvOperator(self.Rd @ self.Wmat)
#         self.Loblique = ( IdentityOperator( (self.n,self.n) ) - (self.W @ self.RdWpinv @ self.Rd) ) @ self.Lpinv
 
#         # Set of coefficient matrix
#         self.C = self.Loblique.T @ self.A @ self.Loblique


#         def _matvec(x):

#             sol, converged = scipy_cg( self.C, self.Loblique.T @ x, x0=None, M=None, *args, **kwargs) 
#             if self.check:
#                 assert converged == 0, "CG algorithm did not converge!"
            
#             # Project onto range of L
#             sol = self.L @ self.Lpinv @ sol
            
#             part_one = self.Loblique @ sol
#             part_two = self.sol_ker_op @ x

#             result = part_one + part_two

#             return result


#         super().__init__( (self.n, self.n), _matvec, _matvec, dtype=np.float64, device="cpu")
















