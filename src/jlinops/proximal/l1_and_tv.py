import numpy as np
from ..derivatives import first_order_derivative_1d, first_order_derivative_2d

from ..util import get_device

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse



def prox_l1_norm(x, lam=1.0):
    """Evaluates prox_f(x) where f(x) = lam*||x||_1.
    This is the soft thresholding function T_lam(x).
    """
    device = get_device(x)
    if device == "cpu":
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    else:
        return cp.sign(x) * cp.maximum(np.abs(x) - lam, 0)



def prox_tv1d_norm(x, lam=1.0, method="fdpg", boundary="none", initialization=None, return_data=False, iterations=20):
    """Evaluates prox_f (x) where f(x) = lam*|| TV(x) ||_1,
    with TV a 1D discrete-gradient matrix.

    Equivalent to the solution of

    min_v  (1/2)*|| v - x ||_2^2 + lam*|| TV(v) ||_1.

    """

    assert method in ["dpg", "fdpg"], "invalid method"
    
    # Build derivative matrix
    n = len(x)
    D, _ = first_order_derivative_1d(n, boundary=boundary) 
    Dt = D.T

    # Objective function
    obj_fn = lambda v: 0.5*(np.linalg.norm(v - x)**2) + lam*np.abs(D @ v).sum()

    # Other
    L = 4

    if method == "dpg":
        
        # Initialize y
        if initialization is None:
            y = np.zeros(D.shape[0])
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
                "obj_vals": np.asarray(obj_vals),
            }
            return v, solver_data
        else:
            return v

    elif method == "fdpg":

        # Initialize
        if initialization is None:
            y_prev = np.zeros(D.shape[0])
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
            t_curr = 0.5*(1 + np.sqrt(1 + L*(t_prev**2)))
            w = y_curr + ((t_prev-1)/t_curr)*(y_curr - y_prev)

            # Advance
            y_prev = y_curr
            t_prev = t_curr
            
            v = (Dt @ y_curr) + x
            obj_vals.append(obj_fn(v))

        if return_data:
            solver_data = {
                "result": v,
                "obj_vals": np.asarray(obj_vals),
            }
            return v, solver_data
        else:
            return v

    else:
        raise NotImplementedError
    



















