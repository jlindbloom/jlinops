
import numpy as np
from numpy.fft import fft as np_fft
from numpy.fft import ifft as np_ifft
from scipy.linalg import circulant as sp_circulant
from scipy.linalg import toeplitz as sp_toeplitz


from .base import _CustomLinearOperator, get_device
from .util import device_to_module


from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupy.fft import fft as cp_fft
    from cupy.fft import ifft as cp_ifft



###################################
#### Random structure matrices ####
###################################


def random_circulant_matrix(n, which="matrix"):
    """Generates a random circulant matrix.
    """
    
    # Draw random column vector
    col = np.random.normal(size=n)

    if which == "matrix":
        return sp_circulant(col)

    elif which == "operator":
        return CirculantOperator(col)

    else:
        raise NotImplementedError



def random_toeplitz_matrix(n, which="matrix", symmetric=True):
    """Generates a random Toeplitz matrix.
    """
    
    # Draw random column vector
    col = np.random.normal(size=n)

    if not symmetric:
        row = np.random.normal(size=n)
        row[0] = col[0]
    else:
        row = None

    if which == "matrix":
        return sp_toeplitz(col, r=row)

    elif which == "operator":
        assert symmetric, "Currently supported only for symmetric matrices."
        return SymmetricToeplitzOperator(col)

    else:
        raise NotImplementedError






####################
#### Embeddings ####
####################


def symmetric_toeplitz_circulant_embedding(T, cval=1.0, which="matrix"):
    """Given a symmetric Toeplitz matrix T, returns a circulant matrix of double the dimensions 
    that contains T on its upper left submatrix.
    """

    # If 1-dimensional, assume we are getting the first column of T
    if T.ndim == 1:
        first_col = T

    # If 2-dimensinoal, assume we are getting the entire matrix for T
    else:
        first_row, first_col = T[0,:], T[:,0]


    C_first_row = np.concatenate( [first_col,  np.concatenate( [ np.atleast_1d(cval) ,   np.flip(first_col[1:]) ])   ]    )
    if which == "matrix":
        C = sp_circulant( C_first_row )
    else:
        C = CirculantOperator( C_first_row )

    return C








#########################
#### LinearOperators ####
#########################
    

class CirculantOperator(_CustomLinearOperator):
    """Represents a circulant LinearOperator.
    """
    def __init__(self, column):

        # Store column
        self.column = column

        # shape
        n = len(self.column)

        # Get device
        device = get_device(self.column)

        if device == "cpu":
            
            self.first_col_fft = np_fft(self.column)
            self.first_col_fft_conj = np.conj(self.first_col_fft)

            def _matvec(v):
                v_fft = np_fft(v)
                result_fft = self.first_col_fft * v_fft  # Element-wise multiplication
                return np.real(np_ifft(result_fft))  # Inverse FFT and take the real part
            
            def _rmatvec(v):
                v_fft = np_fft(v)
                result_fft = self.first_col_fft_conj * v_fft
                return np.real(np_ifft(result_fft))
            
        else:

            self.first_col_fft = cp_fft(self.column)
            self.first_col_fft_conj = np.conj(self.first_col_fft)

            def _matvec(v):
                v_fft = cp_fft(v)
                result_fft = self.first_col_fft * v_fft  # Element-wise multiplication
                return np.real(cp_ifft(result_fft))  # Inverse FFT and take the real part
            
            def _rmatvec(v):
                v_fft = cp_fft(v)
                result_fft = self.first_col_fft_conj * v_fft
                return np.real(cp_ifft(result_fft))

        super().__init__((n,n), _matvec, _rmatvec, device=device)

    def to_gpu(self):
        return CirculantOperator(cp.asarray(self.column))
    
    def to_cpu(self):
        return CirculantOperator(cp.asnumpy(self.column))





class SymmetricToeplitzOperator(_CustomLinearOperator):
    """Represents a symmetric Toeplitz LinearOperator. Matrix-vector products are implemented using a DFT
    using a circulant embedding.

    column: the first column of the Toeplitz operator.
    """
    def __init__(self, column):

        # Shape and device
        device = get_device(column)
        n = len(column)
        
        # Make circulant embedding matrix
        self.C = symmetric_toeplitz_circulant_embedding(column, which="operator")
        
        def _matvec(x):
            x_padded = np.concatenate([x, np.zeros(n)])
            return self.C.matvec(x_padded)[:n]
         

        super().__init__((n,n), _matvec, _matvec, device=device)



















class Block2x2Operator(_CustomLinearOperator):
    """Represents a block 2x2 linear operator as
    A = [A1 & A2 // A3 & A4].
    """
    def __init__(self, A1, A2, A3, A4):

        # Check dimensions for compatibility
        if A1.shape[1] != A3.shape[1] or A2.shape[1] != A4.shape[1] or A1.shape[0] != A2.shape[0] or A3.shape[0] != A4.shape[0]:
            raise ValueError("Block matrices are not compatible")
        
        device = get_device(A1) # get device
        xp = device_to_module(device)

        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4

        # Define the matvec function
        def _matvec(x):
            mid = self.A1.shape[1]
            x1, x2 = x[:mid], x[mid:]
            return xp.hstack([self.A1.matvec(x1) + self.A2.matvec(x2), self.A3.matvec(x1) + self.A4.matvec(x2)])

        # Define the rmatvec function
        def _rmatvec(x):
            mid = self.A1.shape[0]
            x1, x2 = x[:mid], x[mid:]
            return xp.hstack([self.A1.rmatvec(x1) + self.A3.rmatvec(x2), self.A2.rmatvec(x1) + self.A4.rmatvec(x2)])

        shape = (self.A1.shape[0] + self.A3.shape[0], self.A1.shape[1] + self.A2.shape[1])

        super().__init__(shape, _matvec, rmatvec=_rmatvec, device=device)


    def to_gpu(self):

        return Block2x2Operator(self.A1.to_gpu(), self.A2.to_gpu(), self.A3.to_gpu(), self.A4.to_gpu())

    def to_cpu(self):

        return Block2x2Operator(self.A1.to_cpu(), self.A2.to_cpu(), self.A3.to_cpu(), self.A4.to_cpu())






















