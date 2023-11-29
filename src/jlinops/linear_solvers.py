import numpy as np
#from scipy.sparse.linalg import LinearOperator

from .base import _CustomLinearOperator, LinearOperator
from .util import get_device

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp

    

def cg(A, b, x0=None, eps=1e-3, maxits=None):
    """Applies the conjugate gradient method for the solution of A x = b 
    until || A x - b  || / || b || < eps. A should be a linear operator
    representing a SPD or SSPD matrix.

    If maxits is reached but the current solution does not satisfy the termination criteria,
    the current solution is returned and no error is thrown.
    """
    
    # Checks
    assert isinstance(A, LinearOperator), "A must be a jlinops LinearOperator."
    assert A.shape[0] == A.shape[1], "A must be square."
    assert get_device(b) == A.device, "A and b must be on the same device."
    
    # Determine device
    device = A.device
    
    # Figure out shape
    n = A.shape[0]

    # Handle maxits
    if maxits is None:
        maxits = n
    
    if device == "cpu":
    
        # b norm
        bnorm = np.linalg.norm(b)

        # Initialization
        if x0 is None:
            x = np.ones(n)
        else:
            x = x0

        converged = False
        r = b - A.matvec(x)
        d = r.copy()

        its = 0
        for j in range(maxits):

            residual_norm = np.linalg.norm( b - A.matvec( x ) )
            rel_residual_norm = residual_norm/bnorm
            if rel_residual_norm < eps: 
                converged = True
                break

            alpha = (r.T @ r)/(d.T @ A.matvec(d) )
            x = x + alpha*d
            rnew = r - alpha * A.matvec( d )
            beta = (rnew.T @ rnew)/(r.T @ r)
            d = rnew + beta*d
            r = rnew    
            its += 1

    else: 
        
        # b norm
        bnorm = cp.linalg.norm(b)

        # Initialization
        if x0 is None:
            x = cp.ones(n)
        else:
            x = x0

        converged = False
        r = b - A.matvec(x)
        d = r.copy()

        its = 0
        for j in range(maxits):

            residual_norm = cp.linalg.norm( b - A.matvec( x ) )
            rel_residual_norm = residual_norm/bnorm
            if rel_residual_norm < eps: 
                converged = True
                break

            alpha = (r.T @ r)/(d.T @ A.matvec(d) )
            x = x + alpha*d
            rnew = r - alpha * A.matvec( d )
            beta = (rnew.T @ rnew)/(r.T @ r)
            d = rnew + beta*d
            r = rnew    
            its += 1
        

    data = {
        "x": x,
        "n_iterations": its,
        "converged": converged,
    }
    
    return data


















