import numpy as np
import math

from ..derivatives import first_order_derivative_1d, first_order_derivative_2d

from ..util import get_device, get_module
from ..base import MatrixLinearOperator
from ..derivatives import Neumann2D


from .base import ProximalOperator, _to_lam

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse



def prox_l1_norm(x, lam=1.0, rho=None):
    """Evaluates prox_f(x) where f(x) = lam*||x||_1.
    This is the soft thresholding function T_lam(x).
    """
    lam = _to_lam(lam, rho)
    xp = get_module(x)
        
    return xp.sign(x) * xp.maximum(xp.abs(x) - lam, 0)
    


def prox_tv1d_norm(x, lam=1.0, rho=None, method="fdpg", boundary="none", initialization=None, return_data=False, iterations=20):
    """Evaluates prox_f (x) where f(x) = lam*|| TV(x) ||_1,
    with TV a 1D discrete-gradient matrix.

    Equivalent to the solution of

    min_v  (1/2)*|| v - x ||_2^2 + lam*|| TV(v) ||_1.

    """

    assert method in ["dpg", "fdpg"], "invalid method"
    xp = get_module(x)
    device = get_device(x)
    lam = _to_lam(lam, rho)
    
    # Build derivative matrix
    n = len(x)
    D, _ = first_order_derivative_1d(n, boundary=boundary) 
    if device == "gpu":
        D = MatrixLinearOperator(D).to_gpu().A
    Dt = D.T

    # Objective function
    obj_fn = lambda v: 0.5*(xp.linalg.norm(v - x)**2) + lam*xp.abs(D @ v).sum()

    # Other
    L = 4

    if method == "dpg":
        
        # Initialize y
        if initialization is None:
            y = xp.zeros(D.shape[0])
        else:
            y = initialization
        obj_vals = []

        # Iterations
        for j in range(iterations):
            v = (Dt @ y) + x
            Dv = D @ v
            y = y - (1/L)*(Dv) + (1/L)*prox_l1_norm(Dv - L*y, lam=4*lam)
            obj_vals.append(obj_fn(v))

        if return_data:
            solver_data = {
                "result": v,
                "obj_vals": xp.asarray(obj_vals),
            }
            return v, solver_data
        else:
            return v

    elif method == "fdpg":

        # Initialize
        if initialization is None:
            y_prev = xp.zeros(D.shape[0])
        else:
            y_prev = initialization
        w = y_prev.copy()
        t_prev = 1
        obj_vals = []

        # Iterations
        for j in range(iterations):
            u = (Dt @ w) + x
            Du = D @ u
            y_curr = w - (1/L)*Du + (1/L)*prox_l1_norm( Du - L*w, lam=L*lam)
            t_curr = 0.5*(1 + xp.sqrt(1 + L*(t_prev**2)))
            w = y_curr + ((t_prev-1)/t_curr)*(y_curr - y_prev)

            # Advance
            y_prev = y_curr
            t_prev = t_curr
            
            v = (Dt @ y_curr) + x
            obj_vals.append(obj_fn(v))

        if return_data:
            solver_data = {
                "result": v,
                "obj_vals": xp.asarray(obj_vals),
            }
            return v, solver_data
        else:
            return v

    else:
        raise NotImplementedError
    
    
    
    

class ProxTV1DNormOperator(ProximalOperator):
    """Represents prox_f (x) where f(x) = lam*|| TV(x) ||_1,
    with TV a 1D discrete-gradient matrix.

    Equivalent to the solution of

    min_v  (1/2)*|| v - x ||_2^2 + lam*|| TV(v) ||_1.

    """
    
    def __init__(self, n, boundary="none", device="cpu", D=None):
        
        # Build derivative matrix if needed
        self.boundary = boundary
        self.n = n
        if D is None:
            self.D, _ = first_order_derivative_1d(n, boundary=boundary)
            self.D = MatrixLinearOperator(self.D)
            if device == "gpu":
                self.D = self.D.to_gpu()
            self.Dt = self.D.T
        else:
            assert isinstance(D, MatrixLinearOperator), "D must be a MatrixLinearOperator."
            self.D = D
            self.Dt = self.D.T
        
        super().__init__(device=device)
        
        
    def apply(self, x, lam=1.0, rho=None, method="fdpg", initialization=None, return_data=False, iterations=20):
        
        assert method in ["dpg", "fdpg"], "invalid method"
        xp = get_module(x)
        x_device = get_device(x)
        assert x_device == self.device, "x and operator not on same device."
        lam = _to_lam(lam, rho)
        
        # Objective function
        obj_fn = lambda v: 0.5*(xp.linalg.norm(v - x)**2) + lam*xp.abs(self.D @ v).sum()

        # Other
        L = 4

        if method == "dpg":

            # Initialize y
            if initialization is None:
                y = xp.zeros(self.D.shape[0])
            else:
                y = initialization
            obj_vals = []

            # Iterations
            for j in range(iterations):
                v = (self.Dt @ y) + x
                Dv = self.D @ v
                y = y - (1/L)*(Dv) + (1/L)*prox_l1_norm(Dv - L*y, lam=4*lam)
                obj_vals.append(obj_fn(v))

            if return_data:
                solver_data = {
                    "result": v,
                    "obj_vals": xp.asarray(obj_vals),
                }
                return v, solver_data
            else:
                return v

        elif method == "fdpg":

            # Initialize
            if initialization is None:
                y_prev = xp.zeros(self.D.shape[0])
            else:
                y_prev = initialization
            w = y_prev.copy()
            t_prev = 1
            obj_vals = []

            # Iterations
            for j in range(iterations):
                u = (self.Dt @ w) + x
                Du = self.D @ u
                y_curr = w - (1/L)*Du + (1/L)*prox_l1_norm( Du - L*w, lam=L*lam)
                t_curr = 0.5*(1 + xp.sqrt(1 + L*(t_prev**2)))
                w = y_curr + ((t_prev-1)/t_curr)*(y_curr - y_prev)

                # Advance
                y_prev = y_curr
                t_prev = t_curr

                v = (self.Dt @ y_curr) + x
                obj_vals.append(obj_fn(v))

            if return_data:
                solver_data = {
                    "result": v,
                    "obj_vals": xp.asarray(obj_vals),
                }
                return v, solver_data
            else:
                return v

        else:
            raise NotImplementedError


    def to_gpu(self):
        return ProxTV1DNormOperator(self.n, boundary=self.boundary, device="gpu", D=self.D.to_gpu())
    
    
    def to_cpu(self):
        
        return ProxTV1DNormOperator(self.n, boundary=self.boundary, device="cpu", D=self.D.to_cpu())

    
    
    
    
class ProxTVNeumann2DNormOperator(ProximalOperator):
    """Represents prox_f (x) where f(x) = lam*|| TV(x) ||_1,
    with TV a 2D gradient operator with reflexive boundary conditions.

    Equivalent to the solution of

    min_v  (1/2)*|| v - x ||_2^2 + lam*|| TV(v) ||_1.

    """
    
    def __init__(self, grid_shape, device="cpu", D=None):
        
        # Build derivative operator
        self.grid_shape = grid_shape
        self.n = math.prod(grid_shape)
        if D is None:
            self.D = Neumann2D(self.grid_shape, device=device) 
            if device == "gpu":
                self.D = self.D.to_gpu()
            self.Dt = self.D.T
        else:
            self.D = D
            self.Dt = self.D.T
        
        super().__init__(device=device)
        
        
    def apply(self, x, lam=1.0, rho=None, method="fdpg", initialization=None, return_data=False, iterations=20):
        
        assert method in ["dpg", "fdpg"], "invalid method"
        xp = get_module(x)
        x_device = get_device(x)
        assert x_device == self.device, "x and operator not on same device."
        lam = _to_lam(lam, rho)
        
        # Objective function
        obj_fn = lambda v: 0.5*(xp.linalg.norm(v.flatten() - x.flatten())**2) + lam*xp.abs(self.D @ v).sum()

        # Other
        L = 8

        if method == "dpg":

            # Initialize y
            if initialization is None:
                y = xp.zeros(self.D.shape[0])
            else:
                y = initialization
            obj_vals = []

            # Iterations
            for j in range(iterations):
                v = (self.Dt @ y) + x
                Dv = self.D @ v
                y = y - (1/L)*(Dv) + (1/L)*prox_l1_norm(Dv - L*y, lam=4*lam)
                obj_vals.append(obj_fn(v))

            if return_data:
                solver_data = {
                    "result": v,
                    "obj_vals": xp.asarray(obj_vals),
                }
                return v, solver_data
            else:
                return v

        elif method == "fdpg":

            # Initialize
            if initialization is None:
                y_prev = xp.zeros(self.D.shape[0])
            else:
                y_prev = initialization
            w = y_prev.copy()
            t_prev = 1
            obj_vals = []

            # Iterations
            for j in range(iterations):
                u = (self.Dt @ w) + x
                Du = self.D @ u
                y_curr = w - (1/L)*Du + (1/L)*prox_l1_norm( Du - L*w, lam=L*lam)
                t_curr = 0.5*(1 + xp.sqrt(1 + L*(t_prev**2)))
                w = y_curr + ((t_prev-1)/t_curr)*(y_curr - y_prev)

                # Advance
                y_prev = y_curr
                t_prev = t_curr

                v = (self.Dt @ y_curr) + x
                obj_vals.append(obj_fn(v))

            if return_data:
                solver_data = {
                    "result": v,
                    "obj_vals": xp.asarray(obj_vals),
                }
                return v, solver_data
            else:
                return v

        else:
            raise NotImplementedError


    def to_gpu(self):
        
        return ProxTVNeumann2DNormOperator(self.grid_shape, device="gpu", D=self.D.to_gpu())
    
    
    def to_cpu(self):
        
        return ProxTVNeumann2DNormOperator(self.grid_shape, device="cpu", D=self.D.to_cpu())


















