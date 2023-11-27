import numpy as np
from scipy.sparse.linalg._interface import MatrixLinearOperator, _CustomLinearOperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve

from .util import banded_cholesky_factorization



# A note on convention:
# MatrixOperator = jlinop's
# MatrixLinearOperator = scipy's
# BaseLinearOperator = jlinop's
# LinearOperator = scipy's




class MatrixOperator(MatrixLinearOperator):
    """A class for representing linear operators defined by explicit matrices.
    This is a subclass of SciPy's MatrixLinearOperator, which is modified
    such that it retains its identity as a MatrixOperator after various operations
    involving other LinearOperators.
    """


    def __init__(self, A):

        super().__init__(A)


    def _matrix_dot(self, x):
        """Matrix-matrix and matrix-vector multiplication. Essentially the same as
        SciPy's LinearOperator.dot().
        """
        if isinstance(x, MatrixOperator):
            return self.__class__( self.A.dot(x.A) )
        elif isinstance(x, LinearOperator):
            return self.dot(x)
        elif np.isscalar(x):
            return self.__class__(x*self.A)
        else:
            if not issparse(x):
                x = np.asarray(x)
            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)
    


    def _matrix_rdot(self, x):
        """Matrix-matrix and matrix-vector multiplication from the right.
        """
        if isinstance(x, MatrixOperator):
            return self.__class__( x.A.dot(self.A) )
        elif isinstance(x, LinearOperator):
            return self._rdot(x)
        elif np.isscalar(x):
            return self.__class__(x*self.A)
        else:
            if not issparse(x):
                # Sparse matrices shouldn't be converted to numpy arrays.
                x = np.asarray(x)

            # We use transpose instead of rmatvec/rmatmat to avoid
            # unnecessary complex conjugation if possible.
            if x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1:
                return self.T.matvec(x.T).T
            elif x.ndim == 2:
                return self.T.matmat(x.T).T
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)



    def __call__(self, x):
        """On call.
        """
        return self*x
    


    def __mul__(self, x):
        """* multiplication.
        """
        return self._matrix_dot(x)



    def __rmul__(self,x):
        """
        """
        if np.isscalar(x):
            return self.__class__(x*self.A)
        else:
            return self._matrix_rdot(x)



    def __matmul__(self, x):
        """
        """
        if np.isscalar(x):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(x)
    

    
    def __rmatmul__(self, x):
        """
        """
        if np.isscalar(x):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(x)



    def __pow__(self, p):
        """Raise to a power.
        """
        raise NotImplementedError



    def __add__(self, x):
        """Addition.
        """
        if isinstance(x, MatrixOperator):
            return self.__class__(self.A + x.A)
        elif isinstance(x, LinearOperator):
            return self + x
        else:
            return NotImplemented



    def __neg__(self):
        """Negation.
        """
        return -1*self
    


    def __sub__(self, x):
        """Subtraction.
        """
        return self.__add__(-x)



    def _transpose(self):
        #return MatrixOperator(self.A.T)
        return self.__class__(self.A.T)
        
    T = property(_transpose)



    def _adjoint(self):
        return self.__class__(self.A.H)
    
    H = property(_adjoint)



    def _inv(self):
        """Creates a linear operator representing the inverse operator, which uses the LU decomposition.
        """
        return MatrixLUInverseOperator(self)

    Inv = property(_inv)



class MatrixLUInverseOperator(_CustomLinearOperator):
    """Represents the inverse operator of a matrix, where an LU factorization is performed.
    """

    def __init__(self, mat_operator):

        assert isinstance(mat_operator, MatrixLinearOperator), "must be a MatrixOperator."

        # Store the operator
        self.original_op = mat_operator

        # Compute LU decompositon
        self.lu, self.piv = lu_factor(self.original_op.A)

        # Define matvec and rmatvec
        def _matvec(x):
            return lu_solve((self.lu, self.piv), x, trans=0)
        
        def _rmatvec(x):
            return lu_solve((self.lu, self.piv), x, trans=1)
        
        super().__init__(self.original_op.shape, _matvec, _rmatvec)

    

    def _inv(self):
        """Return the original operator if inverse called.
        """
        return self.original_op
    
    Inv = property(_inv)




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
        return MatrixOperator(chol_fac)
    
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












