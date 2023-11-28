import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as splinalg

from ..util import get_device





def banded_cholesky(A, check=False):
    """
    Given a sparse banded matrix :math:`A`, returns the Cholesky factor :math:`L` in the factorizations
    :math:`A = L L^T` with :math:`L` lower-triangular. Here :math:`A` must be a positive definite matrix.
    No permutation is applied.
    """
    
    device = get_device(A)
    if device == "gpu":
        A = A.get()

#     xp = np
#     sp = scipy_sparse
#     splinalg = scipy_splinalg
    
    # Shape
    n = A.shape[0]
    
    # Sparse LU
    LU = splinalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL") 

    # Check for positive-definiteness
    posdef_check = ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all()
    assert posdef_check, "Matrix not positive definite!"
    
    # Extract factor
    L = LU.L.dot( sps.diags(LU.U.diagonal()**0.5) )
    
    # Check?
    if check:
        guess = L @ L.T
        norm_error = np.linalg.norm( guess.toarray() - A.toarray() )
        print(f"L2 norm between L L^T and A: {norm_error}")
    
    return L, LU









