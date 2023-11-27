from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# import scipy.sparse as scipy_sparse
# from scipy.sparse import linalg as scipy_splinalg

from scipy.sparse._base import _spbase as sp_spbase
from scipy.sparse._sputils import isshape as sp_isshape

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.sparse._base import spbase as cp_spbase


# CuPy compatibility
# from .. import CUPY_INSTALLED
# if CUPY_INSTALLED:
#     import cupy as cp
#     import cupy.sparse as cupy_sparse
#     from cupyx.scipy.sparse import linalg as cupy_splinalg



def issparse(x):
    """Checks if a given matrix is a sparse matrix (either SciPy or CuPy).

    Returns:
        bool: whether the matrix is a sparse type or not.

    """
    sp_check = isinstance(x, sp_spbase)
    cp_check = False
    if CUPY_INSTALLED:
        cp_check = isinstance(x, cp_spbase)

    return sp_check or cp_check



def isshape(*args, **kwargs):
    """Is x a tuple of valid dimension? Alias of SciPy's isshape.
    """
    return sp_isshape(*args, **kwargs)



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



def check_adjoint(op, n_rand_vecs=25, tol=1e-1):
    """
    Checks whether the proposed adjoint of a linear operator is really the adjoint.

    Returns:
        bool: whether the adjoint is correct to within the tolerance.
    """

    is_correct_adjoint = True

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

    return is_correct_adjoint



def banded_cholesky_factorization(A, check=False):
    """
    Given a sparse banded matrix :math:`A`, returns the Cholesky factor :math:`L` in the factorizations
    :math:`A = L L^T` with :math:`L` lower-triangular. Here :math:`A` must be a positive definite matrix.
    No permutation is applied.
    """

    xp = np
    sp = scipy_sparse
    splinalg = scipy_splinalg
    
    # Shape
    n = A.shape[0]
    
    # Sparse LU
    LU = splinalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL") 

    # Check for positive-definiteness
    posdef_check = ( LU.perm_r == xp.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all()
    assert posdef_check, "Matrix not positive definite!"
    
    # Extract factor
    L = LU.L.dot( sp.diags(LU.U.diagonal()**0.5) )
    
    # Check?
    if check:
        guess = L @ L.T
        norm_error = xp.linalg.norm( guess.toarray() - A.toarray() )
        print(f"L2 norm between L L^T and A: {norm_error}")
    
    return L, LU



def build_cholesky_factor(matrix_operator):
    """Given a MatrixOperator A for SPD A, builds a dense array L representing L in the factorization A = L L^T with L lower-triangular.
    Computes the decomposition using just the standard Cholesky factorization. Use build_banded_cholesky_factor
    if the matrix is banded.
    """

    # Make sure a MatrixOperator
    assert type(matrix_operator) == MatrixOperator, "Must be a MatrixOperator to perform Cholesky factorization."

    # Extract matrix
    matrix = matrix_operator.A

    # Get Cholesky matrix
    chol_fac = np.linalg.cholesky(matrix)

    return chol_fac



def build_banded_cholesky_factor(matrix_operator, check=False):
    """
    Given a sparse MatrixOperator :math:`A`, returns a sparse matrix representing the Cholesky factor :math:`L` in 
    the factorization :math:`A = L L^T` with :math:`L` lower-triangular. Here :math:`A` must be SPD.
    """
    
    # Make sure a MatrixOperator
    assert type(matrix_operator) == MatrixOperator, "Must be a MatrixOperator to perform Cholesky factorization."
    
    # Extract matrix
    matrix = matrix_operator.A
    
    # Get Cholesky matrix
    chol_fac, LU = banded_cholesky_factorization(matrix, check=check)

    return chol_fac
    





