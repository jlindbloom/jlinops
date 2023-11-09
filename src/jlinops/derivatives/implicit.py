import numpy as np

from scipy.sparse.linalg import LinearOperator


class DiscreteGradientNeumann2D(LinearOperator):
    """Implements a matrix-free operator R representing the anisotropic discrete gradient of an input vector x
    equipped with Neumann boundary conditions. The kernel of this operator is spanned by the constant vector,
    and such that R^T R can be diagonalized using a DCT.
    """
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.M, self.N = self.grid_shape
        super().__init__(shape=(2*self.M*self.N, self.M*self.N), dtype=np.float64)

    def _matvec(self, x):
        
        # Reshape x to have grid shape
        x = x.reshape(self.grid_shape)
        h_diffs, v_diffs = np.zeros(self.grid_shape), np.zeros(self.grid_shape)
        h_diffs[:,:-1] = x[:,1:] - x[:,:-1]
        v_diffs[:-1,:] = x[1:,:] - x[:-1,:]
        output = np.zeros((2,self.M, self.N))
        output[0,:,:] = v_diffs
        output[1,:,:] = h_diffs
        return output.flatten()
    
    def _rmatvec(self, x):
        
        # Reshape x to have (2, M, N) shape
        x = x.reshape((2, self.M, self.N))
        p, q = x[0,:,:].copy(), x[1,:,:]

        # Pad arrays
        q = np.hstack([ np.zeros(self.M)[:,None], q])
        p = np.vstack([ np.zeros(self.N)[None,:], p])

        pdiff = p[1:,:] - p[:-1,:]
        pdiff[-1,:] = - p[-2,:]
        qdiff = q[:,1:] - q[:,:-1]
        qdiff[:,-1] = - q[:,-2]

        # Insert result
        output = -( pdiff + qdiff )

        return output








