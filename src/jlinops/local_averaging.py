import numpy as np
from scipy import signal


from .base import _CustomLinearOperator, LinearOperator
from .util import isshape

# CuPy compatibility
from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    import cupy.sparse as csp
    from cupyx.scipy.signal import convolve as cupy_convolve

    

class LocalAveraging1DOperator(_CustomLinearOperator):
    """
    Represents a 1D "local averaging" operator. returns a smoothed version of
    input where each point represents a local centered average of a ``window_size``-pixel window.
    """

    def __init__(self, n, window_size, device="cpu"):
        
        
        # Shape
        shape = (n,n)
        
        # Bind
        self.grid_shape = grid_shape
        self.window_size = window_size
       
        if device == "cpu":
            
            kernel = np.ones((self.window_size))/self.window_size
            def _matvec_shaped(vec):
                return signal.convolve(vec, kernel, mode="same")
            
        else:
            
            kernel = np.ones((self.window_size))/self.window_size
            def _matvec_shaped(vec):
                return cupy_convolve(vec, kernel, mode="same")
            
        super().__init__(shape, _matvec, _matvec, device=device)
        
        
    def to_gpu(self):
        return LocalAveraging2DOperator(self.grid_shape, self.window_size, device="gpu")
    
    
    def to_cpu(self):
        return LocalAveraging2DOperator(self.grid_shape, self.window_size, device="cpu")
            
        

class LocalAveraging2DOperator(_CustomLinearOperator):
    """
    Represents a 2D "local averaging" operator. Represents a local average within a (``window_size`` x ``window_size``) square 
    about the center pixel.
    """

    def __init__(self, grid_shape, window_size, device="cpu"):
        
        
        # Validate grid_shape
        n = math.prod(grid_shape)
        shape = (n,n)
        assert isshape(grid_shape) and (len(grid_shape) == 2), "Invalid grid_shape."
        
        # Bind
        self.grid_shape = grid_shape
        self.window_size = window_size
       
        if device == "cpu":
            
            kernel = np.ones((window_size,window_size))/(window_size**2)
            def _matvec_shaped(vec):
                return signal.convolve(vec, kernel, mode="same")
            
        else:
            
            kernel = cp.ones((window_size,window_size))/(window_size**2)
            def _matvec_shaped(vec):
                return cupy_convolve(vec, kernel, mode="same")
            
        super().__init__(shape, _matvec, _matvec, device=device)
        
        
    def to_gpu(self):
        return LocalAveraging2DOperator(self.grid_shape, self.window_size, device="gpu")
    
    
    def to_cpu(self):
        return LocalAveraging2DOperator(self.grid_shape, self.window_size, device="cpu")
            
        
        
            
        
        



















# class LocalAveraging2DOperator(LinearOperator):
#     """
#     Represents a "local averaging" operator. In 1D, returns a smoothed version of
#     input where each point represents a local centered average of a ``window_size``-pixel window.
#     In 2D, represents a local average within a (``window_size`` x ``window_size``) square about the center pixel.

#     :param input_shape: the shape of the input.
#     :param window_size: the size of the window.
#     :param boundary_method:
#     """

#     def __init__(
#         self,
#         input_shape,
#         window_size,
#         dtype = np.float64,
#         with_cupy = False,
#     ):

#         # Bind attributes
#         self.input_shape = input_shape
#         self.window_size = window_size

#         # Handle dimension
#         self.ndim = len(self.input_shape)    
#         assert self.ndim in [1, 2], "Only dimensions 1 and 2 supported."

#         # Are we on CPU or GPU
#         if with_cupy:
#             assert CUPY_INSTALLED, "Unavailable, CuPy not installed."
#             xp = cp
#         else:
#             xp = np


#         # Build matvec
#         if self.ndim == 1:
#             kernel = xp.ones((self.window_size))/self.window_size
     
#         elif self.ndim == 2:
#             kernel = xp.ones((window_size,window_size))/(window_size**2)

#         if with_cupy:
#             def _matvec_shaped(vec):
#                 return cupy_convolve(vec, kernel, mode="same", method="fft")
#         else:
#             def _matvec_shaped(vec):
#                 return signal.convolve(vec, kernel, mode="same")

#         super().__init__(self.input_shape, self.input_shape, _matvec_shaped, _matvec_shaped, 
#                         input_dtype=dtype, output_dtype=dtype)
        
        
        
    

      
#     def build_cupy_op(self):
#         """
#         Returns a version of the operator that acts on CuPy arrays.
#         """
        
#         assert CUPY_INSTALLED, "Unavailable, CuPy not installed."
        
        
#         return LocalAveragingOperator(self.input_shape, self.window_size, dtype=cp.float64, with_cupy=True)
        































