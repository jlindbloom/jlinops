import numpy as np
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.sparse.linalg import eigsh as sp_eigsh
from scipy.sparse.linalg import svds as sp_svds
import scipy.sparse as sps

from ..util import get_device
from ..base import MatrixLinearOperator

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    #from cupyx.scipy.sparse.linalg import eigs as cp_eigs
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    from cupyx.scipy.sparse.linalg import svds as cp_svds
    import cupyx.scipy.sparse as csps



def svds(A, *args, **kwargs):

    """Interface to scipy.sparse.linalg.eigs if on CPU and to
    cupyx.scipy.sparse.linalg.eigs if on GPU.
    """

    device = get_device(A)

    if device == "cpu":
        return sp_svds(A, *args, **kwargs)
    else:
        return cp_svds(A, *args, **kwargs)



def fixed_rank_tsvd(A, k=6, flip=False, *args, **kwargs):
    """Given an input LinearOperator A, returns a LinearOperator
    representing the best rank k approximation to A. Computed using
    an iterative method.
    """

    # Compute svd
    U, s, Vh = svds(A, k=k)

    # Flip the order?
    if flip:
        U = np.flip(U, axis=1)
        s = np.flip(s)
        Vh = np.flip(Vh, axis=0)

    # Build LinearOperator for diagonal part
    device = A.device
    if device == "cpu":

        S = sps.diags(s, shape=(k,k))

    else:

        S = csps.diags(s, shape=(k,k))

    # Build low rank operator
    Ak = MatrixLinearOperator(U) @ ( MatrixLinearOperator(S) @ MatrixLinearOperator(Vh) )

    return Ak, (U, s, Vh)    



def variable_rank_tsvd(A):
    """Computes terms in the truncated SVD until a stopping criterion is met.
    """
    raise NotImplementedError



def eigs(A, *args, **kwargs):
    """Interface to scipy.sparse.linalg.eigs if on CPU and to
    cupyx.scipy.sparse.linalg.eigs if on GPU.
    """

    device = get_device(A)

    if device == "cpu":
        return sp_eigs(A, *args, **kwargs)
    else:
        raise NotImplementedError
        #return cp_eigs(A, *args, **kwargs)



def eigsh(A, all=False, flip=False, *args, **kwargs):
    """Interface to scipy.sparse.linalg.eigsh if on CPU and to
    cupyx.scipy.sparse.linalg.eigsh if on GPU.
    """

    device = get_device(A)

    if not all:
        if device == "cpu":
            S, V = sp_eigsh(A, *args, **kwargs)
            if flip:
                print("flipped")
                return np.flip(S), np.flip(V, axis=1)
            return S, V
        else:
            S, V = cp_eigsh(A, *args, **kwargs)
            if flip:
                print("flipped")
                return cp.flip(S), np.flip(V, axis=1)
            return S, V

    else:

        n = A.shape[1]
        if device == "cpu":
            S2, V2 = sp_eigsh(A, which="LM", k=n-1)
            S1, V1 = sp_eigsh(A, which="SM", k=1)
            V = np.hstack([V1, V2])
            S = np.hstack([S1, S2])
            if flip:
                return np.flip(S), np.flip(V, axis=1)
            return S, V
        else:
            S2, V2 = cp_eigsh(A, which="LM", k=n-1)
            S1, V1 = cp_eigsh(A, which="SM", k=1)
            V = cp.hstack([V1, V2])
            S = cp.hstack([S1, S2])
            if flip:
                return cp.flip(S), cp.flip(V, axis=1)
            return S, V












