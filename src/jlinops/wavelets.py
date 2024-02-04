import numpy as np

import pywt

from .base import _CustomLinearOperator




class Wavelet1DOperator(_CustomLinearOperator):
    """Represents a 1D wavelet operator extracting both approximation and detail coefficients.
    Can only handle even-shaped inputs for now.
    """
    def __init__(self, in_shape, wavelet='db1', mode='symmetric'):
        
        self.wavelet = wavelet
        self.mode = mode
        device = "cpu"
        

        def _matvec(x):

            # Perform forward wavelet transform
            cA, cD = pywt.dwt(x, "db1")

            # Flatten the coefficients
            return np.hstack([cA, cD])
        
        out_shape = len(_matvec(np.ones(in_shape)))
        coeff_len = pywt.dwt_coeff_len(in_shape, pywt.Wavelet(self.wavelet), mode=self.mode)
        print(out_shape)
        
        def _rmatvec(x):

            cA, cD = x[:coeff_len], x[coeff_len:]

            return pywt.idwt(cA, cD, wavelet=self.wavelet, mode=self.mode)

        
        shape = (out_shape, in_shape)

            
        super().__init__( shape, _matvec, _rmatvec, device=device, dtype=np.float64)


    def to_cpu(self):
        raise NotImplementedError
    
    def to_gpu(self):
        raise NotImplementedError





class WaveletDetail1DOperator(_CustomLinearOperator):
    """Represents a 1D wavelet operator extracting only the detail coefficients.
    Can only handle even-shaped inputs for now.
    """

    def __init__(self, in_shape, wavelet='db1', mode='symmetric'):
        
        self.wavelet = wavelet
        self.mode = mode
        device = "cpu"
        

        def _matvec(x):

            # Perform forward wavelet transform
            cA, cD = pywt.dwt(x, "db1")

            # Flatten the coefficients
            return cD
        
        out_shape = len(_matvec(np.ones(in_shape)))
        
        def _rmatvec(x):


            return pywt.idwt(None, x, wavelet=self.wavelet, mode=self.mode)

        
        shape = (out_shape, in_shape)

            
        super().__init__( shape, _matvec, _rmatvec, device=device, dtype=np.float64)


    def to_cpu(self):
        raise NotImplementedError
    
    def to_gpu(self):
        raise NotImplementedError