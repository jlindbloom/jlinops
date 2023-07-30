import numpy as np
from scipy.sparse.linalg._interface import _CustomLinearOperator
from scipy.ndimage import gaussian_filter







class Gaussian1DBlurOperator(_CustomLinearOperator):
    """Implements a Gaussian blurring operator for 1D vectors.
    """

    def __init__(self, n, blur_sigma=1.0, mode="wrap"):
        """
        n: the size of the 1D input vector.
        blur_sigma: controls the spread of the blurring kernel.
        mode: how to handle the boundary.
        """

        # Make sure boundary condition mode is valid
        valid_modes = ["wrap", "reflect", "constant", "nearest", "mirror"]
        assert mode in valid_modes, f"Invalid mode, must be one of {valid_modes}."
        self.mode = mode

        # Bind attribute
        self.blur_sigma = blur_sigma 

        # Define matvec and rmatvec
        def _matvec(x):
            return gaussian_filter(x, float(self.blur_sigma), mode=self.mode)


        super().__init__( (n, n), _matvec, _matvec)



class Gaussian2DBlurOperator(_CustomLinearOperator):
    """Implements a Gaussian blurring operator for 1D vectors.
    """

    def __init__(self, shape, blur_sigma=1.0, mode="wrap"):
        """
        shape: a list-like containing the shape of an input vector (shaped).
        blur_sigma: controls the spread of the blurring kernel.
        mode: how to handle the boundary.
        """

        # Make sure boundary condition mode is valid
        valid_modes = ["wrap", "reflect", "constant", "nearest", "mirror"]
        assert mode in valid_modes, f"Invalid mode, must be one of {valid_modes}."
        self.mode = mode

        # Bind attribute
        self.blur_sigma = blur_sigma 

        try:
            m, n = shape
        except:
            raise Exception("Invalid shape.")

        # Also make shapes
        self.input_shape = (m,n)
        self.output_shape = (m,n)

        # Define matvec and rmatvec
        def _matvec(x):
            return gaussian_filter(x.reshape((m,n)), float(self.blur_sigma), mode=self.mode).flatten()


        super().__init__( (n, n), _matvec, _matvec)



    def matvec_shaped(self, x):
        """
        """
        return gaussian_filter(x, float=(self.blur_sigma), mode=self.mode )



    def rmatvec_shaped(self, x):
        """
        """
        return self.matvec_shaped(x)

