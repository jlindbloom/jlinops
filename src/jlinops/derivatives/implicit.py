import numpy as np

from ..base import _CustomLinearOperator

from .. import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp

    

class Neumann2D(_CustomLinearOperator):
    """Implements a matrix-free operator R representing the anisotropic discrete gradient of an input vector x
    equipped with Neumann boundary conditions. The null space of this operator is spanned by the constant vector,
    and is such that R^T R can be diagonalized using a 2-dimensional DCT.
    """
    def __init__(self, grid_shape, device="cpu"):
        
        # Handle grid shape
        self.grid_shape = grid_shape
        m, n = grid_shape
        shape = (2*m*n, m*n)
        self.M, self.N = self.grid_shape
        
        # Define matvec and rmatvec
        if device == "cpu":
            
            def _matvec(x):

                # Reshape x to have grid shape
                x = x.reshape(self.grid_shape)
                h_diffs, v_diffs = np.zeros(self.grid_shape), np.zeros(self.grid_shape)
                h_diffs[:,:-1] = x[:,1:] - x[:,:-1]
                v_diffs[:-1,:] = x[1:,:] - x[:-1,:]
                output = np.zeros((2,m,n))
                output[0,:,:] = v_diffs
                output[1,:,:] = h_diffs
                return output.flatten()
            
            def _rmatvec(x):
        
                # Reshape x to have (2, M, N) shape
                x = x.reshape((2, m, n))
                p, q = x[0,:,:], x[1,:,:]

                # Pad arrays
                q = np.hstack([ np.zeros(m)[:,None], q])
                p = np.vstack([ np.zeros(n)[None,:], p])

                pdiff = p[1:,:] - p[:-1,:]
                pdiff[-1,:] = - p[-2,:]
                qdiff = q[:,1:] - q[:,:-1]
                qdiff[:,-1] = - q[:,-2]

                # Insert result
                output = -( pdiff + qdiff )

                return output

            
        else:
            
            def _matvec(x):

                # Reshape x to have grid shape
                x = x.reshape(self.grid_shape)
                h_diffs, v_diffs = cp.zeros(self.grid_shape), cp.zeros(self.grid_shape)
                h_diffs[:,:-1] = x[:,1:] - x[:,:-1]
                v_diffs[:-1,:] = x[1:,:] - x[:-1,:]
                output = cp.zeros((2,m,n))
                output[0,:,:] = v_diffs
                output[1,:,:] = h_diffs
                return output.flatten()
            
            def _rmatvec(x):
        
                # Reshape x to have (2, M, N) shape
                x = x.reshape((2, m, n))
                p, q = x[0,:,:], x[1,:,:]

                # Pad arrays
                q = cp.hstack([ np.zeros(m)[:,None], q])
                p = cp.vstack([ np.zeros(n)[None,:], p])

                pdiff = p[1:,:] - p[:-1,:]
                pdiff[-1,:] = - p[-2,:]
                qdiff = q[:,1:] - q[:,:-1]
                qdiff[:,-1] = - q[:,-2]

                # Insert result
                output = -( pdiff + qdiff )

                return output

        
        super().__init__(shape, _matvec, _rmatvec, device=device)

        
    def to_gpu(self):
        return Neumann2D(self.grid_shape, device="gpu")
    
    def to_cpu(self):
        return Neumann2D(self.grid_shape, device="cpu")
        






class Dirichlet2D(_CustomLinearOperator):
    """Implements a matrix-free operator R representing the anisotropic discrete gradient of an input vector x
    equipped with Neumann boundary conditions. The null space of this operator is spanned by the constant vector,
    and is such that R^T R can be diagonalized using a 2-dimensional DCT.
    """
    def __init__(self, grid_shape, device="cpu"):
        
        # Handle grid shape
        self.grid_shape = grid_shape
        m, n = grid_shape
        shape = (2*m*n, m*n)
        self.M, self.N = self.grid_shape

        if device == "cpu":

            def matvec(x):
                # Reshape the vector into a 2D grid
                grid = x.reshape(self.M, self.N)

                # Compute the x-derivative
                dx = np.zeros_like(grid)
                dx[:-1, :] = grid[1:, :] - grid[:-1, :]
                dx[-1, :] = -grid[-1,:]

                # Compute the y-derivative
                dy = np.zeros_like(grid)
                dy[:, :-1] = grid[:, 1:] - grid[:, :-1]
                dy[:, -1] = -grid[:, -1]

                # Flatten and combine the derivatives
                return np.hstack((dx.ravel(), dy.ravel()))

            def rmatvec(y):
                # Reshape the vector into two 2D grids for dx and dy
                dx, dy = np.split(y, 2)
                dx = dx.reshape(self.M, self.N)
                dy = dy.reshape(self.M, self.N)

                # Compute the transpose operations with sign correction
                dxt = np.zeros_like(dx)
                dyt = np.zeros_like(dy)

                # Transpose operation for x-derivative
                dxt[1:, :] -= dx[:-1, :]
                dxt[:, :] += dx[:, :]

                # Transpose operation for y-derivative
                dyt[:, 1:] -= dy[:, :-1]
                dyt[:, :] += dy[:, :]


                # Combine, flatten, and apply sign correction
                return -(dxt + dyt).ravel()

        else:
    
            def matvec(x):
                # Reshape the vector into a 2D grid
                grid = x.reshape(self.M, self.N)

                # Compute the x-derivative
                dx = cp.zeros_like(grid)
                dx[:-1, :] = grid[1:, :] - grid[:-1, :]
                dx[-1, :] = -grid[-1,:]

                # Compute the y-derivative
                dy = np.zeros_like(grid)
                dy[:, :-1] = grid[:, 1:] - grid[:, :-1]
                dy[:, -1] = -grid[:, -1]

                # Flatten and combine the derivatives
                return cp.hstack((dx.ravel(), dy.ravel()))

            def rmatvec(y):
                # Reshape the vector into two 2D grids for dx and dy
                dx, dy = cp.split(y, 2)
                dx = dx.reshape(self.M, self.N)
                dy = dy.reshape(self.M, self.N)

                # Compute the transpose operations with sign correction
                dxt = cp.zeros_like(dx)
                dyt = cp.zeros_like(dy)

                # Transpose operation for x-derivative
                dxt[1:, :] -= dx[:-1, :]
                dxt[:, :] += dx[:, :]

                # Transpose operation for y-derivative
                dyt[:, 1:] -= dy[:, :-1]
                dyt[:, :] += dy[:, :]

                # Combine, flatten, and apply sign correction
                return -(dxt + dyt).ravel()


        super().__init__(shape, matvec, rmatvec, device=device)


    def to_gpu(self):
        return Dirichlet2D(self.grid_shape, device="gpu")

    def to_cpu(self):
        return Dirichlet2D(self.grid_shape, device="cpu")









class Dirichlet2DSym(_CustomLinearOperator):
    """Implements a matrix-free operator R representing the anisotropic discrete gradient of an input vector x
    equipped with Dirichlet boundary conditions. The null space of this operator is spanned by the constant vector,
    and is symmetrized such that R^T R can be diagonalized using a 2-dimensional DCT.
    """
    def __init__(self, grid_shape, device="cpu"):
        
        # Handle grid shape
        self.grid_shape = grid_shape
        m, n = grid_shape
        shape = (2*m*n + m + n, m*n)
        self.M, self.N = self.grid_shape

        if device == "cpu":

            def matvec(x):

                # Reshape the vector into a 2D grid
                grid = x.reshape(self.M, self.N)

                # Compute the x-derivative
                dx = np.zeros((self.M+1, self.N))
                dx[1:-1, :] = grid[1:, :] - grid[:-1, :]
                dx[-1, :] = - grid[-1,:]
                dx[0, :] = grid[0,:]

                # Compute the y-derivative
                dy = np.zeros((self.M, self.N+1))
                dy[:, 1:-1] = grid[:, 1:] - grid[:, :-1]
                dy[:, -1] = -grid[:, -1]
                dy[:, 0] = grid[:, 0]

                # Flatten and combine the derivatives
                return np.hstack((dx.ravel(), dy.ravel()))

            def rmatvec(y):
                
                # The length of the x-derivative and y-derivative vectors
                len_dx = (self.M + 1) * self.N
                len_dy = self.M * (self.N + 1)
                
                # Extract the derivatives from y
                dx = y[:len_dx].reshape((self.M + 1, self.N))
                dy = y[len_dx:len_dx + len_dy].reshape((self.M, self.N + 1))
                
                # Compute the negative divergence from dx and dy
                # Initialize divergence grid with zeros
                div = np.zeros((self.M, self.N))
                
                # Compute divergence from the x-derivative
                div[:-1, :] -= dx[1:-1, :]
                div[1:, :] += dx[1:-1, :]
                div[-1, :] -= dx[-1, :]
                div[0, :] += dx[0, :]
                
                # Compute divergence from the y-derivative
                div[:, :-1] -= dy[:, 1:-1]
                div[:, 1:] += dy[:, 1:-1]
                div[:, -1] -= dy[:, -1]
                div[:, 0] += dy[:, 0]

                # Return the flattened divergence
                return div.ravel()

        else:
    
            def matvec(x):

                # Reshape the vector into a 2D grid
                grid = x.reshape(self.M, self.N)

                # Compute the x-derivative
                dx = cp.zeros((self.M+1, self.N))
                dx[1:-1, :] = grid[1:, :] - grid[:-1, :]
                dx[-1, :] = - grid[-1,:]
                dx[0, :] = grid[0,:]

                # Compute the y-derivative
                dy = cp.zeros((self.M, self.N+1))
                dy[:, 1:-1] = grid[:, 1:] - grid[:, :-1]
                dy[:, -1] = -grid[:, -1]
                dy[:, 0] = grid[:, 0]

                # Flatten and combine the derivatives
                return cp.hstack((dx.ravel(), dy.ravel()))
            

            def rmatvec(y):
                
                # The length of the x-derivative and y-derivative vectors
                len_dx = (self.M + 1) * self.N
                len_dy = self.M * (self.N + 1)
                
                # Extract the derivatives from y
                dx = y[:len_dx].reshape((self.M + 1, self.N))
                dy = y[len_dx:len_dx + len_dy].reshape((self.M, self.N + 1))
                
                # Compute the negative divergence from dx and dy
                # Initialize divergence grid with zeros
                div = cp.zeros((self.M, self.N))
                
                # Compute divergence from the x-derivative
                div[:-1, :] -= dx[1:-1, :]
                div[1:, :] += dx[1:-1, :]
                div[-1, :] -= dx[-1, :]
                div[0, :] += dx[0, :]
                
                # Compute divergence from the y-derivative
                div[:, :-1] -= dy[:, 1:-1]
                div[:, 1:] += dy[:, 1:-1]
                div[:, -1] -= dy[:, -1]
                div[:, 0] += dy[:, 0]

                # Return the flattened divergence
                return div.ravel()
                

        super().__init__(shape, matvec, rmatvec, device=device)


    def to_gpu(self):
        return Dirichlet2DSym(self.grid_shape, device="gpu")

    def to_cpu(self):
        return Dirichlet2DSym(self.grid_shape, device="cpu")























# Legacy

# class DiscreteGradientNeumann2D(LinearOperator):
#     """Implements a matrix-free operator R representing the anisotropic discrete gradient of an input vector x
#     equipped with Neumann boundary conditions. The kernel of this operator is spanned by the constant vector,
#     and such that R^T R can be diagonalized using a DCT.
#     """
#     def __init__(self, grid_shape):
#         self.grid_shape = grid_shape
#         self.M, self.N = self.grid_shape
#         super().__init__(shape=(2*self.M*self.N, self.M*self.N), dtype=np.float64)

#     def _matvec(self, x):
        
#         # Reshape x to have grid shape
#         x = x.reshape(self.grid_shape)
#         h_diffs, v_diffs = np.zeros(self.grid_shape), np.zeros(self.grid_shape)
#         h_diffs[:,:-1] = x[:,1:] - x[:,:-1]
#         v_diffs[:-1,:] = x[1:,:] - x[:-1,:]
#         output = np.zeros((2,self.M, self.N))
#         output[0,:,:] = v_diffs
#         output[1,:,:] = h_diffs
#         return output.flatten()
    
#     def _rmatvec(self, x):
        
#         # Reshape x to have (2, M, N) shape
#         x = x.reshape((2, self.M, self.N))
#         p, q = x[0,:,:].copy(), x[1,:,:]

#         # Pad arrays
#         q = np.hstack([ np.zeros(self.M)[:,None], q])
#         p = np.vstack([ np.zeros(self.N)[None,:], p])

#         pdiff = p[1:,:] - p[:-1,:]
#         pdiff[-1,:] = - p[-2,:]
#         qdiff = q[:,1:] - q[:,:-1]
#         qdiff[:,-1] = - q[:,-2]

#         # Insert result
#         output = -( pdiff + qdiff )

#         return output








