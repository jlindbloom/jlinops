import numpy as np


def _to_lam(lam, rho):
    """For handling different conventions for proximal operator notation.
    """
    assert not ( (lam is not None) and (rho is not None) ), "Only 1 of lam/rho can be passed."
    assert not ( (lam is None) and (rho is None) ), "Must pass one of lam/rho."
    if lam is not None:
        return lam
    else:
        lam = rho**2
        return lam
    

class ProximalOperator:
    """Base class for representing a proximal operator.
    """

    def __init__(self, device="cpu"):
        
        self.device= device


    def apply(self, x, lam=1.0, rho=None):
        
        raise NotImplementedError
    
    def to_gpu(self):
        
        raise NotImplementederror
        
    def to_cpu(self):
        
        raise NotImplementedError












