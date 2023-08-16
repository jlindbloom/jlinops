from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._interface import _CustomLinearOperator



class DiagonalizedOperator(_CustomLinearOperator):
    """Represents a linear operator that has been diagonalized as
        A = P D P^H,
    for which D is known and P is known as a linear operator.
    """
    def __init__(self, P, eigenvalues):

        self.eigenvalues = eigenvalues
        self.P = P

        # Define matvec and rmatvec
        def _matvec(x):
            tmp = self.P.H @ x
            tmp = self.eigenvalues*tmp
            tmp = self.P @ tmp
            return tmp
        
        def _rmatvec(x):
            tmp = self.P.H @ x
            tmp = self.eigenvalues.conj()*tmp
            tmp = self.P @ tmp
            return tmp
        
        super().__init__( self.P.shape, _matvec, _rmatvec )



