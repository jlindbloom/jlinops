import numpy as np

from ..util import get_module, device_to_module
from ..stacked import StackedOperator
from ..base import IdentityOperator
from ..pseudoinverse import QRPinvOperator



def cgls(A, b, x0=None, maxiter=None, return_search_vectors=False, 
         return_iterates=False, tol=1e-3, relative=True, early_stopping=True, **kwargs):
    """CGLS method applied to solution of
    min_x || A x - b ||_2.
    """
    
    device = A.device
    xp = get_module(b)
    n = A.shape[1]
    
    # Initialization
    if x0 is None:
        x = xp.zeros(n)
    else:
        x = x0
    
    # Norm of b, for relative stopping criteria
    bnorm = xp.linalg.norm(b)
        
    # Maxiter
    if maxiter is None:
        maxiter = n
    
    r_prev = b - A.matvec(x)
    At_r_prev = A.rmatvec(r_prev)
    d = A.rmatvec(r_prev)
    
    init_residual_norm = xp.linalg.norm(r_prev)
    At_residual_norm = xp.linalg.norm(At_r_prev)
    residual_norms = [init_residual_norm]
    At_residual_norms = [At_residual_norm]
    x_norms = [xp.linalg.norm(x)]
    n_iters = 0
    
    if return_search_vectors:
        search_vectors = [d]
    if return_iterates:
        iterates = [x]
        
    # Check if stopping criterion already satisfied?
    stopping_criteria_satisfied = False
    if early_stopping:
        if relative:
            if (At_residual_norm/bnorm) < tol:
                stopping_criteria_satisfied = True
        else:
            if At_residual_norm < tol:
                stopping_criteria_satisfied = True

    if stopping_criteria_satisfied:
        data = {
            "x": x,
            "residual_norms": xp.asarray(residual_norms),
            "x_norms": xp.asarray(x_norms),
            "n_iters": n_iters,
            "iterates": None,
            "search_vectors": None,
            "converged": stopping_criteria_satisfied,
            "At_residual_norms": xp.asarray(At_residual_norms),
        }
        return data
    
    else:
        
        # Do iterations
        for j in range(maxiter):

            # Compute iterate
            A_d = A.matvec(d) # only matvec with A
            alpha = (xp.linalg.norm(At_r_prev)/xp.linalg.norm(A_d))**2
            x = x + alpha*d
            r_curr = r_prev - alpha*A_d
            At_r_curr = A.rmatvec(r_curr) # only matvec with At
            beta = (xp.linalg.norm(At_r_curr)/xp.linalg.norm(At_r_prev))**2
            d = At_r_curr + beta*d

            # Advance
            r_prev = r_curr
            At_r_prev = At_r_curr
            n_iters += 1

            # Store search vectors and iterates?
            if return_search_vectors:
                search_vectors.append(d)
            if return_iterates:
                iterates.append(x)

            # Track residual and solution norm
            residual_norm = xp.linalg.norm(r_prev)
            At_residual_norm = xp.linalg.norm(At_r_prev) # This is the one that is going to zero
            residual_norms.append(residual_norm)
            At_residual_norms.append(At_residual_norm)
            x_norms.append(xp.linalg.norm(x))

            # Stopping criteria?
            if early_stopping:
                if relative:
                    if (At_residual_norm/bnorm) < tol:
                        stopping_criteria_satisfied = True
                        break
                else:
                    if At_residual_norm < tol:
                        stopping_criteria_satisfied = True
                        break

        data = {
            "x": x,
            "residual_norms": xp.asarray(residual_norms),
            "x_norms": xp.asarray(x_norms),
            "n_iters": n_iters,
            "iterates": None,
            "search_vectors": None,
            "converged": stopping_criteria_satisfied,
            "At_residual_norms": xp.asarray(At_residual_norms),
        }

        if return_search_vectors:
            data["search_vectors"] = xp.vstack(search_vectors).T

        if return_iterates:
            data["iterates"] = xp.vstack(iterates).T

        return data



def rlstsq(A, b, lam=1.0, shift=None, initialization=None, *args, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| x - shift ||_2^2
    using a CGLS method.
    """
    # Build Atilde
    device = A.device
    xp = device_to_module(device)
    n = A.shape[1]
    Atilde = StackedOperator([A, xp.sqrt(lam)*IdentityOperator((n,n), device=device)])
    if shift is None:
        shift = xp.zeros(n)
    rhs = xp.hstack([b, xp.sqrt(lam)*shift])
    data = cgls(Atilde, rhs, x0=initialization, *args, **kwargs)
    return data



def trlstsq(A, R, b, lam=1.0, shift=None, initialization=None, *args, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| R x - shift ||_2^2
    using a CGLS method. It is assumed that
    null(A) and null(R) intersect trivially.
    """
    # Build Atilde
    device = A.device
    xp = device_to_module(device)
    n = A.shape[1]
    Atilde = StackedOperator([A, np.sqrt(lam)*R])
    zeros = xp.zeros(R.shape[0])
    
    if shift is None:
        shift = xp.zeros(R.shape[0])
    
    # If R is none, assume we mean identity (so really, not transformed at all)
    if R is None:
        R = IdentityOperator((n,n), device=device)
        
    rhs = xp.hstack([b, xp.sqrt(lam)*shift])
    data = cgls(Atilde, rhs, x0=initialization, *args, **kwargs)
    
    return data



def trlstsq_rinv(A, Rinv, b, lam=1.0, shift=None, initialization=None, R=None, *args, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| R x - shift ||_2^2
    using a transformation to standard form + a CGLS method. 
    It is assumed that null(A) and null(R) intersect trivially,
    and additionally that R is square invertible.
    """
    # Build Atilde
    device = A.device
    xp = device_to_module(device)
    n = A.shape[1]
    
    # Build Atilde, handle initialization, and solve
    Atilde = A @ Rinv 
    if initialization is not None:
        assert R is not None, "must provide R."
        initialization = R.matvec(initialization) # transform initialization to z coordinate
        
    data = rlstsq(Atilde, b, shift=shift, lam=lam, initialization=initialization, *args, **kwargs)
    
    # Transform solution and overwrite cgls_data
    data["z"] = data["x"].copy() # z is new coordinate
    x = Rinv @ data["x"]
    data["x"] = x
    
    return data



def trlstsq_rtker(A, Rpinv, b, lam=1.0, shift=None, chol_fac=False, R=None, initialization=None, *args, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| R x - shift ||_2^2
    using a transformation to standard form + a CGLS method. 
    It is assumed that null(A) and null(R) intersect trivially,
    and additionally that R has a trivial kernel (but possibly
    non-square).
    
    If chol_fac=False, Rpinv should represent the MP pseudoinverse.
    If chol_fac=True, Rpinv should represent square L^-T where R^T R = L L^T.
    """
    # Build Atilde
    device = A.device
    xp = device_to_module(device)
    n = A.shape[1]
    
    Atilde = A @ Rpinv 
    if shift is not None:
        assert R is not None, "R must be provided."
        if not chol_fac:
            shift = R @ (Rpinv @ shift)
        else:
            shift = Rpinv.rmatvec(R.rmatvec(shift))
            
    if initialization is not None:
        assert R is not None, "must provide R."
        if not chol_fac:
            initialization = R.matvec(initialization) # transform initialization to z coordinate
        else:
            initialization = Linv.rmatvec(R.rmatvec(R.matvec(initialization)))
        
    data = rlstsq(Atilde, b, lam=lam, shift=shift, *args, **kwargs)
    
    # Transform solution and overwrite cgls_data
    data["z"] = data["x"].copy() # z is new coordinate
    x = Rpinv @ data["x"]
    data["x"] = x
    
    return data



def trlstsq_rntker(A, Rpinv, W, b, lam=1.0, AWpinv=None, shift=None, R=None, initialization=None, *args, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| R x - shift ||_2^2
    using a transformation to standard form + a CGLS method. 
    It is assumed that null(A) and null(R) intersect trivially,
    and additionally that R has a nontrivial kernel.
    
    W should be a MatrixLinearOperator whose columns span null(R).
    """
    # Build Atilde
    device = A.device
    xp = device_to_module(device)
    n = A.shape[1]
    
    # Build oblique pseudoinverse
    if AWpinv is None:
        AWpinv = QRPinvOperator( A.matmat(W.A) )
    oblique_pinv = ( IdentityOperator((n,n)) -  W @ (AWpinv @ A)  ) @ Rpinv
    
    # Get contribution from kernel
    x_null = W @ (AWpinv @ b)
    
    # Get contribution from complement
    Atilde = A @ oblique_pinv
    if shift is not None:
        assert R is not None, "R must be provided."
        shift = R @ (Rpinv @ shift)
        
    if initialization is not None:
        assert R is not None, "R must be provided."
        #initialization = oblique_pinv.matvec(R.matvec(initialization))
        Wpinv = QRPinvOperator( W.A )
        initialization = initialization - (W @ ( Wpinv @ initialization))
        initialization = R.matvec(initialization)
    
    data = rlstsq(Atilde, b, lam=lam, shift=shift, initialization=initialization, *args, **kwargs)
    
    
    # Transform solution and overwrite cgls_data
    data["z"] = data["x"].copy() # z is transformed coordinate
    data["x_null"] = x_null
    x_null_comp = oblique_pinv @ data["x"]
    data["x_null_comp"] = x_null_comp
    x = x_null_comp + x_null
    data["x"] = x
    data["oblique_pinv"] = oblique_pinv
    
    return data




def trlstsq_standard_form(A, b, Rinv=None, Rpinv=None, W=None, chol_fac=None, **kwargs):
    """Solves the regularized least-squares problem
    min_x || A x - b ||_2^2 + lam*|| R x - shift ||_2^2
    using a transformation to standard form + a CGLS method.
    
    This function wraps the standard form solvers for the different
    cases dependent on R. These include:
    
    R invertible: trlstsq_rinv
    R not invertible but trivial kernel: trlstsq_rtker
    R has trivial kernel: trlstsq_rntker
    """
    
    # assert not ( (R is not None) and (Rinv is None) and (Rpinv is None) and (W is None) ), "not enough information to solve in standard form."
    
    if (Rinv is None) and (Rpinv is None) and (W is None):
        # Assume we mean identity regularization, so already in standard form
        return rlstsq(A, b, **kwargs)
    elif (Rinv is not None) and (Rpinv is None) and (W is None):
        # R invertible case
        return trlstsq_rinv(A, Rinv, b, **kwargs)
    elif (Rinv is None) and (Rpinv is not None) and (W is None):
        # R trivial kernel case
        return trlstsq_rtker(A, Rpinv, b, **kwargs)
    elif (Rinv is None) and (Rpinv is not None) and (W is not None):
        # R nontrivial kernel case
        return trlstsq_rntker(A, Rpinv, W, b, **kwargs)
    else:
        raise ValueError("Invalid combination for standard form solver.")
    
    
    


