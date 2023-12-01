from .pseudoinverse import QRPinvOperator
from .base import IdentityOperator, MatrixLinearOperator
from .util import check_adjoint



def build_oblique_pinv(X, Ypinv, W, XWpinv=None, check=False):
    """Returns a linear operator representing the oblique pseudoinverse relative to X and Y.
    The kernels of X and Y must intersect trivially.
    
    X: a LinearOperator.
    Y: another LinearOperator.
    Ypinv: a LinearOperator representing the Moore-Penrose pseudoinverse of Y.
    W: a MatrixLinearOperator such that col(W) = null(Y).
    
    """
    
    # If XWpinv not passed, build using QR pseudoinverse method
    if XWpinv is None:
        XWpinv = QRPinvOperator( MatrixLinearOperator(X.dot(W.A)) )
    
    # Build op
    n = X.shape[1]
    identity_op = IdentityOperator( (n,n) ) 
    oblique_pinv = (identity_op - (W @ (XWpinv @ X ) ) ) @ Ypinv
    
    # Check if necessary
    if check:
        assert check_adjoint(oblique_pinv)
    
    return oblique_pinv