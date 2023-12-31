import numpy as np
import math

import scipy.sparse as sps



def first_order_derivative_1d(N, boundary="none"):
    """Constructs a sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions. Also returns a dense matrix W
    whose column span the nullspace of the operator (if trivial, W = None).
    """
    
    assert boundary in ["none", "periodic", "zero", "reflexive"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=1)
    #d_mat = sps.csc_matrix(d_mat)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[-1,0] = -1
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "zero":
        W = None
        pass
    elif boundary == "none":
        d_mat = d_mat[:-1,:]
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "reflexive":
        d_mat[-1,-1] = 0
        W = np.atleast_2d(np.ones(N)).T
    else:
        pass
    
    return d_mat, W



def first_order_derivative_2d_split(grid_shape, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an two-dimensional array.

    Returns: (R1, W1), (R2, W2)

    R1 extracts horizontal gradients, R2 extracts vertical gradients.
    col(W1) = span(null(R1)), col(W2) = span(null(R2)).
    """
    
    M, N = grid_shape

    # Construct our differencing matrices
    R1, W1 = first_order_derivative_1d(N, boundary=boundary)
    R2, W2 = first_order_derivative_1d(M, boundary=boundary)
    
    # Take kronecker products
    eye_vert = sps.eye(M)
    R1 = sps.kron(R1, eye_vert)
    if W1 is not None:
        W1 = sps.kron(W1, eye_vert)
    
    eye_horiz = sps.eye(N)
    R2 = sps.kron(eye_horiz, R2)
    if W2 is not None:
        W2 = sps.kron(eye_horiz, W2)
    
    return (R1, W1), (R2, W2)



def first_order_derivative_2d(grid_shape, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
    
    Returns: (R, W)

    R contains R1 (horizontal gradient) stacked on top of R2 (vertical gradient).
    col(W) = span(null(R)).
    """

    (R1, W1), (R2, W2) = first_order_derivative_2d_split(grid_shape, boundary=boundary)

    R = sps.vstack([R1, R2])

    if boundary in ["none", "periodic", "reflexive"]:
        W = np.atleast_2d(np.ones(R.shape[1])).T
    elif boundary == "zero":
        W = None
    else:
        raise NotImplementedError
    
    return R, W









# def build_2d_first_order_derivative_2d(shape, boundary="periodic"):
#     """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
#     Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
#     to compute matrix-vector product. First set is horizontal gradient, second is vertical.
#     """
    
#     M, N = shape

#     # Construct our differencing matrices
#     d_mat_horiz = build_1d_first_order_derivative(N, boundary=boundary)
#     d_mat_vert = build_1d_first_order_derivative(M, boundary=boundary)
#     d_mat_horiz = d_mat_horiz.A
#     d_mat_vert = d_mat_vert.A
    
#     # Build the combined matrix
#     eye_vert = sps.eye(M)
#     d_mat_one = sps.kron(d_mat_horiz, eye_vert)
    
#     eye_horiz = sps.eye(N)
#     d_mat_two = sps.kron(eye_horiz, d_mat_vert)
    
#     return SparseMatrixOperator(d_mat_one), MatrixOperator(d_mat_two)



# def build_2d_first_order_derivative(shape, boundary="periodic"):
#     """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
#     Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
#     to compute matrix-vector product. First set is horizontal gradient, second is vertical.
#     """

#     d_mat_one, d_mat_two = build_2d_first_order_derivative_split(shape, boundary=boundary)

#     full_diff_mat = sps.vstack([d_mat_one.A, d_mat_two.A])
    
#     return SparseMatrixOperator(full_diff_mat)

