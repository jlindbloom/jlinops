import numpy as np
import matplotlib.pyplot as plt


# CuPy compatibility
# from .. import CUPY_INSTALLED
# if CUPY_INSTALLED:
#     import cupy as cp
#     import cupy.sparse as cupy_sparse
#     from cupyx.scipy.sparse import linalg as cupy_splinalg

import scipy.sparse as scipy_sparse
from scipy.sparse import linalg as scipy_splinalg


from .matrix import MatrixOperator






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
    """Given a MatrixOperator A for SPD A, builds a MatrixOperator L representing L in the factorization A = L L^T with L lower-triangular.
    Computes the decomposition using just the standard Cholesky factorization. Use build_banded_cholesky_factor
    if the matrix is banded.
    """

    # Make sure a MatrixOperator
    assert type(matrix_operator) == MatrixOperator, "Must be a MatrixOperator to perform Cholesky factorization."

    # Extract matrix
    matrix = matrix_operator.A

    # Get Cholesky matrix
    chol_fac = np.linalg.cholesky(matrix)

    return MatrixOperator(chol_fac)



def build_banded_cholesky_factor(matrix_operator, check=False):
    """
    Given a sparse MatrixOperator :math:`A`, returns a MatrixOperator representing the Cholesky factor :math:`L` in 
    the factorization :math:`A = L L^T` with :math:`L` lower-triangular. Here :math:`A` must be SPD.
    """
    
    # Make sure a MatrixOperator
    assert type(matrix_operator) == MatrixOperator, "Must be a MatrixOperator to perform Cholesky factorization."
    
    # Extract matrix
    matrix = matrix_operator.A
    
    # Get Cholesky matrix
    chol_fac, LU = banded_cholesky_factorization(matrix, check=check)

    return MatrixOperator(chol_fac)
    





