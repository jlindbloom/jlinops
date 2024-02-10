import numpy as np
import math

import scipy.sparse as sps



def first_order_derivative_1d(N, boundary="none"):
    """Constructs a sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions. Also returns a dense matrix W
    whose column span the nullspace of the operator (if trivial, W = None).
    """
    
    assert boundary in ["none", "periodic", "zero", "reflexive", "zero_sym"], "Invalid boundary parameter."
    
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
    elif boundary == "zero_sym":
        d_mat = sps.csc_matrix(d_mat)
        new_row = sps.csc_matrix(np.zeros(d_mat.shape[1]))
        d_mat = sps.vstack([new_row, d_mat])
        d_mat[0,0] = -1
        W = None
    else:
        pass
    
    return d_mat, W



def second_order_derivative_1d(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none"], "Invalid boundary parameter."
    
    d_mat = -sps.eye(N)
    d_mat.setdiag(2,k=1)
    d_mat.setdiag(-1,k=2)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[-2,0] = -1
        d_mat[-1,0] = 2
        d_mat[-1,1] = -1
        raise NotImplementedError
    elif boundary == "zero":
        raise NotImplementedError
    elif boundary == "none":
        d_mat = d_mat[:-2, :]
        w1 = np.ones(N)/np.linalg.norm(np.ones(N))
        w2 = np.arange(N)/np.linalg.norm(np.arange(N))
        W = np.vstack([w1, w2]).T
    else:
        pass
    
    return d_mat, W



def third_order_derivative_1d(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none"], "Invalid boundary parameter."
    
    d_mat = -sps.eye(N)
    d_mat.setdiag(3,k=1)
    d_mat.setdiag(-3,k=2)
    d_mat.setdiag(1,k=3)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        raise NotImplementedError
    elif boundary == "zero":
        raise NotImplementedError
    elif boundary == "none":
        d_mat = d_mat[:-3, :]
        w1 = np.ones(N)/np.linalg.norm(np.ones(N))
        w2 = np.arange(N)/np.linalg.norm(np.arange(N))
        w3 = np.cumsum(w2)/np.linalg.norm(np.cumsum(w2))
        W = np.vstack([w1, w2, w3]).T
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









def build_neumann2d_sparse_matrix(grid_shape):
     """Makes a sparse matrix corresponding to the matrix-free Neumann2D operator.
     """

     m, n = grid_shape

     Rv, _ = first_order_derivative_1d(m, boundary="reflexive")
     Rv *= -1.0

     Rh, _ = first_order_derivative_1d(n, boundary="reflexive")
     Rh *= -1.0

     return sps.vstack([sps.kron(Rv, sps.eye(n)), sps.kron(sps.eye(m), Rh) ])



def get_neumann2d_laplacian_diagonal(grid_shape):
    """Returns a vector containing the diagonal of the Neumann2D laplacian.
    """

    m, n = grid_shape
    d = 4*np.ones(math.prod(grid_shape))

    # Inner 3's
    d[ n : -n : n ] -= 1
    d[ 2*n - 1 : -n : n ] -= 1

    # First and last 2's, and 3's
    d[0] = 2
    d[-1] = 2
    d[n-1] = 2
    d[-n] = 2
    d[1:n-1] = 3
    d[-n+1:-1] = 3

    return d



def get_neumann2d_laplacian_tridiagonal(grid_shape):
    """Returns a vector containing the tridiagonal of the Neumann2D laplacian.
    """
    d = get_neumann2d_laplacian_diagonal(grid_shape)
    m, n = grid_shape
    l = -1*np.ones(math.prod(grid_shape) - 1)
    l[n-1::n] = 0

    return l.copy(), d, l.copy()




def build_dirichlet2d_sparse_matrix(grid_shape):
    """Makes a sparse matrix corresponding to the matrix-free Dirichlet2D operator.
    """

    Q, _ = first_order_derivative_2d(grid_shape, boundary="zero")

    return Q






def build_dirichlet2dsym_sparse_matrix(grid_shape):
     """Makes a sparse matrix corresponding to the matrix-free Dirichlet2DSym operator.
     """

     m, n = grid_shape

     Rv, _ = first_order_derivative_1d(m, boundary="zero_sym")
     Rv *= -1.0

     Rh, _ = first_order_derivative_1d(n, boundary="zero_sym")
     Rh *= -1.0

     return sps.vstack([sps.kron(Rv, sps.eye(n)), sps.kron(sps.eye(m), Rh) ])










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

