import numpy as np

from scipy.sparse import issparse, isspmatrix_coo, coo_matrix
import scipy.sparse as sps



def bandwidth(matrix):
    """
    Calculates the bandwidth of an input matrix.
    """
    if issparse(matrix):
        if not isspmatrix_coo(matrix):
            matrix = coo_matrix(matrix)
        distances = abs(matrix.row - matrix.col)
        bandwidth = distances.max()
    elif isinstance(matrix, np.ndarray):
        rows, cols = matrix.shape
        max_distance = 0
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] != 0:
                    distance = abs(i - j)
                    max_distance = max(max_distance, distance)
        bandwidth = max_distance
    else:
        raise NotImplementedError
    
    return bandwidth



def random_lower_triangular(n, k):
    """
    Create an n x n lower triangular sparse matrix with bandwidth k,
    where non-zero entries are from the standard normal distribution.
    
    Parameters:
    n (int): The size of the matrix.
    k (int): The bandwidth of the matrix.
    
    Returns:
    scipy.sparse.csr_matrix: An n x n lower triangular sparse matrix.
    """
    row_indices = []
    col_indices = []
    data = []
    
    # Generate the non-zero entries within the specified bandwidth
    for i in range(n):
        for j in range(max(0, i - k + 1), i + 1):
            row_indices.append(i)
            col_indices.append(j)
            data.append(np.random.normal())
    
    # Create a COO matrix
    mat = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    
    # Convert to CSR format for efficiency
    mat = mat.tocsr()
    
    return mat




def extract_tridiag(A):
    """Returns the tridiagonal of an input matrix.
    """

    l = A.diagonal(-1)
    d = A.diagonal()
    u = A.diagonal(1)

    return l, d, u


def build_tridiag(l, d, u):
    """Returns a tridiagonal matrix from input diagonals.
    """

    return sps.diags([l, d, u], offsets=(-1, 0, 1))




# def random_lower_triangular(n, k):
#     """
#     Create an n x n lower triangular matrix with bandwidth k, 
#     where non-zero entries are from the standard normal distribution.
    
#     Parameters:
#     n (int): The size of the matrix.
#     k (int): The bandwidth of the matrix.
    
#     Returns:
#     np.ndarray: An n x n lower triangular matrix.
#     """
#     # Initialize an n x n matrix of zeros
#     matrix = np.zeros((n, n))
    
#     # Fill the matrix within the specified bandwidth
#     for i in range(n):
#         for j in range(max(0, i - k + 1), i + 1):
#             matrix[i, j] = np.random.normal()
    
#     return matrix