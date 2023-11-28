from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import operator

# import scipy.sparse as scipy_sparse
# from scipy.sparse import linalg as scipy_splinalg

from scipy.sparse._base import _spbase as sp_spbase
from scipy.sparse._sputils import isshape as sp_isshape
from scipy.sparse._sputils import asmatrix as sp_asmatrix


from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse._base import spmatrix as cp_spmatrix


# CuPy compatibility
# from .. import CUPY_INSTALLED
# if CUPY_INSTALLED:
#     import cupy as cp
#     import cupy.sparse as cupy_sparse
#     from cupyx.scipy.sparse import linalg as cupy_splinalg



##################
### Basic util ###
##################


def get_device(x):
    """Determines whether input array x is on cpu/gpu.
    """
    if not CUPY_INSTALLED:
        return "cpu"
    else:
        module = cp.get_array_module(x)
        if module == cp:
            return "gpu"
        else:
            return "cpu"


        
def issparse(x):
    """Checks if a given matrix is a sparse matrix (either SciPy or CuPy).

    Returns:
        bool: whether the matrix is a sparse type or not.

    """
    sp_check = isinstance(x, sp_spbase)
    cp_check = False
    if CUPY_INSTALLED:
        cp_check = isinstance(x, cp_spmatrix)

    return sp_check or cp_check



# def isshape(*args, **kwargs):
#     """Is x a tuple of valid dimension? Alias of SciPy's isshape.
#     """
#     return sp_isshape(*args, **kwargs)



def isshape(x, nonneg=False, allow_ndim=False) -> bool:
    """Is x a valid tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    If allow_ndim, shapes of any dimensionality are allowed.
    """
    ndim = len(x)
    if not allow_ndim and ndim != 2:
        return False
    for d in x:
        if not isintlike(d):
            return False
        if nonneg and d < 0:
            return False
    return True



def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    device = get_device(x)
    
    if device == "cpu":
        if np.ndim(x) != 0:
            return False
    else:
        if cp.ndim(x) != 0:
            return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            msg = "Inexact indices into sparse matrices are not allowed"
            raise ValueError(msg)
        return loose_int
    return True



def asmatrix(data, dtype=None):
    """Turns array into matrix type.
    """
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        return data
    elif CUPY_INSTALLED and isinstance(data, cp.ndarray):
        return cp.asarray(data, dtype=dtype).view()
    else:
        return np.asarray(data, dtype=dtype).view(np.matrix)
    


def check_adjoint(op, n_rand_vecs=25, tol=1e-1):
    """
    Checks whether the proposed adjoint of a linear operator is really the adjoint.

    Returns:
        bool: whether the adjoint is correct to within the tolerance.
    """

    is_correct_adjoint = True
    
    if op.device == "cpu":
        for j in range(n_rand_vecs):

            x = np.random.normal(size=op.shape[1]).flatten() # x
            y = np.random.normal(size=op.shape[0]).flatten() # y

            tilde_y = op.matvec(x) # \tilde{y} = A x
            tilde_x = op.rmatvec(y) # \tilde{x} = A y

            dot_x = np.dot(tilde_x, x) # dot product of x's
            dot_y = np.dot(y, np.real(tilde_y)) # dot product of y's

            # Check: these two dot products should be the same
            if abs( dot_x - dot_y ) > tol:
                is_correct_adjoint = False
    
    else:
        for j in range(n_rand_vecs):

            x = cp.random.normal(size=op.shape[1]).flatten() # x
            y = cp.random.normal(size=op.shape[0]).flatten() # y

            tilde_y = op.matvec(x) # \tilde{y} = A x
            tilde_x = op.rmatvec(y) # \tilde{x} = A y

            dot_x = cp.dot(tilde_x, x) # dot product of x's
            dot_y = cp.dot(y, np.real(tilde_y)) # dot product of y's

            # Check: these two dot products should be the same
            if abs( dot_x - dot_y ) > tol:
                is_correct_adjoint = False
                
    return is_correct_adjoint




