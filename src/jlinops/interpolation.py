import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix

from .base import _CustomLinearOperator



def nearest_1d_grid_idx(x, grid, a=None, b=None):
    """Given an array of x-coordinates, returns the index of the nearest x-cordinate 
    (from the left) on the grid.
    """
    if a is None: a = np.amin(grid)
    if b is None: b = np.amax(grid)
    
    assert np.all(np.diff(x) >= 0), "Array must be ordered in increasing order."
    assert np.amin(x) > a, "query points must be in between interval endpoints."
    assert np.amax(x) < b, "query points must be in between interval endpoints."

    indices = np.searchsorted(grid, x, side='right') - 1
    
    return indices




class LinearNonuniform2Uniform1DGriddingOperator(_CustomLinearOperator):
    """Represents a linear operator mapping y-coordinates on a regular grid
    to y-coordinates on an irregular grid, using piecewise linear interpolation. 
    Used for the inverse interpolation problem.

    interval: a tuple [a,b] defining the uniform grid interval.
    n_points: the number of points on the uniform grid.
    x_nonuniform: the x-coordinates of the nonuniform grid.
    """
    
    def __init__(self, interval, n_points, x_nonuniform):

        self.interval = interval
        self.a, self.b = interval
        self.n_points = n_points
        self.x_uniform = np.linspace(self.a, self.b, self.n_points)
        self.h = self.x_uniform[1] - self.x_uniform[0]
        self.x_nonuniform = x_nonuniform

        # Get coefficients for linear interpolation
        self.left_idxs = nearest_1d_grid_idx(self.x_nonuniform, self.x_uniform, a=self.a, b=self.b)
        left_uniform_grid_points = self.x_uniform[self.left_idxs]
        self.linear_interp_coeffs = ((self.x_nonuniform-left_uniform_grid_points)/self.h)

        # Build regridding matrix
        shape = ( len(self.x_nonuniform), len(self.x_uniform) )
        regridding_mat = lil_matrix(shape)

        # Fill matrix with coefficients
        for i in range(len(self.x_nonuniform)):
            # Figure out column index
            j = self.left_idxs[i]

            # Get coeff
            c = self.linear_interp_coeffs[i]

            # Fill matrix
            regridding_mat[i,j] = (1-c)
            regridding_mat[i,j+1] = c

        self.regridding_mat = regridding_mat
        self.regridding_mat_t = self.regridding_mat.T
        
        def _matvec(v):
             return self.regridding_mat @ v

        def _rmatvec(v):
             return self.regridding_mat_t @ v
        
        super().__init__(shape, matvec=_matvec, rmatvec=_rmatvec, device="cpu")


    






