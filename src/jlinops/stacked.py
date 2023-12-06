import numpy as np


from .base import _CustomLinearOperator
from .util import split_array, device_to_module



class StackedOperator(_CustomLinearOperator):
    """Represents a 'stacked' LinearOperator.
    
    opertors: a list of compatible LinearOperators.
    """

    def __init__(self, operators):
     
        self.operators = operators
        
        # Checks
        n = self.operators[0].shape[1]
        m = self.operators[0].shape[0]
        device = self.operators[0].device
        self._lengths = [m]
        for op in self.operators[1:]:
            assert n == op.shape[1], "not all operators have same input shape."
            assert device == op.device, "not all operators on same device."
            mk = op.shape[0]
            self._lengths.append(mk)
            m += mk
            
        shape = (m, n)
        xp = device_to_module(device)
        
        def _matvec(x):
            evals = []
            for op in self.operators:
                evals.append(op.matvec(x))
            return xp.hstack(evals)
                        
        def _rmatvec(x):
            evals = []
            for j, subset in enumerate(split_array(x, self._lengths)):
                evals.append(self.operators[j].rmatvec(subset))
            return xp.sum(evals, axis=0)
        
        super().__init__(shape, _matvec, _rmatvec, device=device)
        
        
    def to_gpu(self):
        gpu_ops = []
        for op in self.operators:
            gpu_ops.append(op.to_gpu())
        return StackedOperator(gpu_ops)
    
    def to_cpu(self):
        cpu_ops = []
        for op in self.operators:
            cpu_ops.append(op.to_cpu())
        return StackedOperator(cpu_ops)








