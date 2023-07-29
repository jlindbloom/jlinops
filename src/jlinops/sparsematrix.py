import numpy as np
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator
from scipy.sparse.linalg import splu

from .matrix import MatrixOperator
from .cholesky import banded_cholesky_factorization


class SparseMatrixOperator(MatrixOperator):
    """Represents a SciPy sparse matrix.
    """

    def __init__(self, A):

        super().__init__(A)



    def _inv(self):
        """Return the inverse operator.
        """
        return SparseMatrixLUInverseOperator(self)
    
    Inv = property(_inv)



class SparseMatrixLUInverseOperator(_CustomLinearOperator):
    """Represents the inverse operator of a matrix, where an LU factorization is performed.
    """

    def __init__(self, mat_operator):
        
        # Store the original operator
        self.original_op = mat_operator

        # Perform LU decomposition
        self.lu = splu(self.original_op.A)


        # Define matvec and rmatvec
        def _matvec(x):
             return self.lu.solve(x, trans="N")
        
        def _rmatvec(x):
            return self.lu.solve(x, trans="T")

        super().__init__(self.original_op.shape, _matvec, _rmatvec)



class SparseBandedSPDMatrixOperator(SparseMatrixOperator):
    """Represents a banded SPD sparse matrix. The banded part is important,
    as we do NOT try to permute the rows/columns to minimize bandwidth when factorizing.
    """

    def __init__(self, A):

        super().__init__(A)



    def _inv(self):
        
        return SparseBandedCholInvSPDMatrixOperator(self)
    
    Inv = property(_inv)



    def _chol(self):
        chol_fac, _ = banded_cholesky_factorization(self.A)
        return chol_fac
    
    Chol = property(_chol)
    


class SparseBandedCholInvSPDMatrixOperator(_CustomLinearOperator):
    """Represents the inverse operator of a sparse SPD matrix,
    computed using the LU decomposition. Also binds Cholesky
    factor as an attribute.
    """

    def __init__(self, mat_operator):

        # Store original operator
        self.original_op = mat_operator

        # Compute cholesky
        self.chol_fac, self.LU = banded_cholesky_factorization(self.original_op.mat)

        # Define matvec and rmatvec
        def _matvec(x):
            return self.LU.solve(x, trans="N")
        
        def _rmatvec(x):
            return self.LU.solve(x, trans="T")
        
        super().__init__(self.original_op.shape, _matvec, _rmatvec)
    


    def _inv(self):
        return self.original_op
    
    Inv = property(_inv)







