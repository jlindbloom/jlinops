import numpy as np
from scipy.sparse.linalg import LinearOperator




def cg(A, b, x0=None, eps=1e-3, maxits=None):
    """Applies the conjugate gradient method for the solution of A x = b 
    until || A x - b  || / || b || < eps. A must be SPD.

    If maxits is reached but the current solution does not satisfy the termination criteria,
    the current solution is returned and no error is thrown.
    """
    
    # Figure out shape
    n = A.shape[0]

    # Handle maxits
    if maxits is None:
        maxits = n
    
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
    

    data = {
        "x": x,
        "n_iterations": its,
    }
    
    return data


















