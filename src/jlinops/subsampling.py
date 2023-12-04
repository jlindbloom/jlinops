import numpy as np
import math

from .base import _CustomLinearOperator
from .util import get_device

from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    
    

class Subsampling1DOperator(_CustomLinearOperator):
    """Represents a subsampling operator.
    """

    def __init__(self, mask):
        """
        mask: an array representing a binary mask for subsampling, where 1 means observed and 0 means unobserved.
        """
        
        # Determine device
        device = get_device(mask)
        
        # Validate the mask
        assert mask.ndim == 1, "Mask must be 1-dimensional."
        
        if device == "cpu":
            assert np.all((mask == 0) | (mask == 1)), "Array must contain only 0 and 1."
        else:
            assert cp.all((mask == 0) | (mask == 1)), "Array must contain only 0 and 1."
                
        # Make ints
        if device == "cpu":
            mask = mask.astype(np.int32)
        else:
            mask = mask.astype(cp.int32)
        
        # Bind mask
        self.mask = mask
        
        # Determine shape
        n = len(mask)
        num_nz = self.mask.sum()
        shape = (num_nz, n)

        # Other
        self.idxs_observed = np.where(self.mask == 1)

        if device == "cpu":
        
            # Define matvec and rmatvec
            def _matvec(x):
                return x[self.idxs_observed]

            def _rmatvec(x):
                tmp = np.zeros(n)
                tmp[self.idxs_observed] = np.squeeze(x)
                return tmp
        else:
            
            # Define matvec and rmatvec
            def _matvec(x):
                return x[self.idxs_observed]

            def _rmatvec(x):
                tmp = cp.zeros(n)
                tmp[self.idxs_observed] = cp.squeeze(x)
                return tmp
            
            
        super().__init__( shape, _matvec, _rmatvec, device=device)
        
        
    def to_gpu(self):
        return Subsampling1DOperator(cp.asarray(self.mask))
    
    def to_cpu(self):
        return Subsampling1DOperator(cp.asnumpy(self.mask))




    
class Subsampling2DOperator(_CustomLinearOperator):
    
    def __init__(self, mask):
        
        # Determine device
        device = get_device(mask)
        
        # Validate the mask
        assert mask.ndim == 2, "Mask must be 2-dimensional."
        m, n = mask.shape
        
        if device == "cpu":
            assert np.all((mask == 0) | (mask == 1)), "Array must contain only 0 and 1."
        else:
            assert cp.all((mask == 0) | (mask == 1)), "Array must contain only 0 and 1."
        
        # Make ints
        if device == "cpu":
            mask = mask.astype(np.int32)
        else:
            mask = mask.astype(cp.int32)
            
        # Bind mask
        self.mask = mask
        self.mask_flat = self.mask.flatten()
        
        # Determine shape
        n = math.prod(mask.shape)
        if device == "cpu":
            num_nz = self.mask.sum()
        else:
            num_nz = (cp.asnumpy(self.mask)).sum()
        shape = (num_nz, n)
        
        # Other
        self.idxs_observed = np.where(self.mask_flat == 1)

        if device == "cpu":
        
            # Define matvec and rmatvec
            def _matvec(x):
                return x[self.idxs_observed]

            def _rmatvec(x):
                tmp = np.zeros(n)
                tmp[self.idxs_observed] = x
                return tmp
        else:
            
            # Define matvec and rmatvec
            def _matvec(x):
                return x[self.idxs_observed]

            def _rmatvec(x):
                tmp = cp.zeros(n)
                tmp[self.idxs_observed] = x
                return tmp
            
            
        super().__init__( shape, _matvec, _rmatvec, device=device, dtype=None)
        
        
    def to_gpu(self):
        
        return Subsampling2DOperator(cp.asarray(self.mask))
    
    
    def to_cpu(self):
        
        return Subsampling2DOperator(cp.asnumpy(self.mask))