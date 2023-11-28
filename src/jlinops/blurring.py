import numpy as np
from scipy.ndimage import gaussian_filter
import math

from .base import _CustomLinearOperator
from .util import isshape

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter 



class Gaussian1DBlurOperator(_CustomLinearOperator):
    """Implements a Gaussian blurring operator for 1D vectors.
    """

    def __init__(self, n, blur_sigma=1.0, mode="wrap", device="cpu"):
        """
        n: the size of the 1D input vector.
        blur_sigma: controls the spread of the blurring kernel.
        mode: how to handle the boundary.
        """

        # Make sure boundary condition mode is valid
        valid_modes = ["wrap", "reflect", "constant", "nearest", "mirror"]
        assert mode in valid_modes, f"Invalid mode, must be one of {valid_modes}."
        
        # Bind
        self.mode = mode
        self.blur_sigma = blur_sigma 
        
        if device == "cpu":
            def _matvec(x):
                return gaussian_filter(x, float(self.blur_sigma), mode=self.mode)
        else:
            def _matvec(x):
                return cupy_gaussian_filter(x, float(self.blur_sigma), mode=self.mode)


        super().__init__( (n, n), _matvec, _matvec, device=device)
        
        
    def to_gpu(self):
        return Gaussian1DBlurOperator( self.shape[0], blur_sigma=self.blur_sigma,
                                     mode=self.mode, device="gpu")
        
    def to_cpu(self):
        return Gaussian1DBlurOperator( self.shape[0], blur_sigma=self.blur_sigma,
                                     mode=self.mode, device="cpu")



class Gaussian2DBlurOperator(_CustomLinearOperator):
    """Implements a Gaussian blurring operator for 2D vectors.
    """

    def __init__(self, grid_shape, blur_sigma=1.0, mode="wrap", device="cpu"):
        """
        shape: a list-like containing the shape of an input vector (shaped).
        blur_sigma: controls the spread of the blurring kernel.
        mode: how to handle the boundary.
        """

        # Make sure boundary condition mode is valid
        valid_modes = ["wrap", "reflect", "constant", "nearest", "mirror"]
        assert mode in valid_modes, f"Invalid mode, must be one of {valid_modes}."
        
        # Bind
        self.mode = mode
        self.blur_sigma = blur_sigma 
        self.grid_shape = grid_shape
        
        # Shape is (n,n), must multiply grid dimensions
        n = math.prod(grid_shape)
        shape = (n,n)
        assert isshape(grid_shape) and (len(grid_shape) == 2), "Invalid grid_shape."
       
        if device == "cpu":
            
            def _matvec(x):
                return gaussian_filter(x.reshape(self.grid_shape), float(self.blur_sigma), mode=self.mode).flatten()
            
        else:
           
            def _matvec(x):
                return cupy_gaussian_filter(x.reshape(self.grid_shape), float(self.blur_sigma), mode=self.mode).flatten()
            

        super().__init__( (n, n), _matvec, _matvec, device=device)

    def to_gpu(self):
        return Gaussian2DBlurOperator(self.grid_shape, blur_sigma=self.blur_sigma, mode=self.mode, device="gpu")
        
        
    def to_cpu(self):
        return Gaussian2DBlurOperator(self.grid_shape, blur_sigma=self.blur_sigma, mode=self.mode, device="cpu")
        
        
    def matvec_shaped(self, x):
        """Applies the matvec to a shaped input, returning a shaped output.
        """
        assert x.shape == self.grid_shape, "Invalid shape for input x."
        return self.matvec(x.flatten()).reshape(self.grid_shape)


    def rmatvec_shaped(self, x):
        """
        """
        assert x.shape == self.grid_shape, "Invalid shape for input x."
        return self.matvec_shaped(x)

