import numpy as np
import math

import scipy.sparse as sps

from ..matrix import MatrixOperator
from ..sparsematrix import SparseMatrixOperator



def build_1d_first_order_derivative(N, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none", "periodic", "zero"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=-1)
    #d_mat = sps.csc_matrix(d_mat)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[0,-1] = -1
    elif boundary == "zero":
        pass
    elif boundary == "none":
        d_mat = d_mat[1:,:]
    else:
        pass
    
    return SparseMatrixOperator(d_mat)



def build_2d_first_order_derivative_split(shape, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
    Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
    to compute matrix-vector product. First set is horizontal gradient, second is vertical.
    """
    
    M, N = shape

    # Construct our differencing matrices
    d_mat_horiz = build_1d_first_order_derivative(N, boundary=boundary)
    d_mat_vert = build_1d_first_order_derivative(M, boundary=boundary)
    d_mat_horiz = d_mat_horiz.A
    d_mat_vert = d_mat_vert.A
    
    # Build the combined matrix
    eye_vert = sps.eye(M)
    d_mat_one = sps.kron(d_mat_horiz, eye_vert)
    
    eye_horiz = sps.eye(N)
    d_mat_two = sps.kron(eye_horiz, d_mat_vert)
    
    return SparseMatrixOperator(d_mat_one), MatrixOperator(d_mat_two)



def build_2d_first_order_derivative(shape, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
    Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
    to compute matrix-vector product. First set is horizontal gradient, second is vertical.
    """

    d_mat_one, d_mat_two = build_2d_first_order_derivative_split(shape, boundary=boundary)

    full_diff_mat = sps.vstack([d_mat_one.A, d_mat_two.A])
    
    return SparseMatrixOperator(full_diff_mat)

