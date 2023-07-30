import numpy as np
from scipy.sparse.linalg._interface import _CustomLinearOperator




class SubsamplingOperator(_CustomLinearOperator):
    """Represents a subsampling operator.
    """

    def __init__(self, mask):
        """
        mask: an array representing a binary mask for subsampling, where 1 means observed and 0 means unobserved.
        """

        # Bind mask
        self.mask = mask
        
        # Figure out input and output shapes
        input_shape = self.mask.size
        num_nz = self.mask.sum()
        output_shape = num_nz

        # Other
        self.idxs_observed = np.where(self.mask == 1)

        # Define matvec and rmatvec
        def _matvec(x):
            return x[self.idxs_observed]
        
        def _rmatvec(x):
            tmp = np.zeros(input_shape)
            tmp[self.idxs_observed] = x
            return tmp

        super().__init__( (output_shape, input_shape), _matvec, _rmatvec)


