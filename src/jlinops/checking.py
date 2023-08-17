import numpy as np




def check_adjoint(op, n_rand_vecs=25, tol=1e-1):
    """
    Checks whether the adjoint of a linear operator is really the adjoint.
    """

    is_correct_adjoint = True

    for j in range(n_rand_vecs):

        x = np.random.normal(size=op.shape[1]).flatten() # x
        y = np.random.normal(size=op.shape[0]).flatten() # y

        tilde_y = op.matvec(x) # \tilde{y} = A x
        tilde_x = op.rmatvec(y) # \tilde{x} = A y

        dot_x = np.dot(tilde_x, x) # dot product of x's
        dot_y = np.dot(y, np.real(tilde_y)) # dot product of y's
        
        # Check: these two dot products should be the same
        if abs( dot_x - dot_y ) > tol:
            is_correct_adjoint = False

    return is_correct_adjoint









