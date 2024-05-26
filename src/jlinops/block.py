import numpy as np

from .base import _CustomLinearOperator




class BlockDiagonalOperator(_CustomLinearOperator):
    """Given square operators A_1, \ldots, A_p of shapes n_i \times n_i, represents the block diagonal operator diag(A_1, \ldots, A_p).
    """
    def __init__(self, As, device="cpu", type=1):

        # Setup
        self.As = As # list of operators
        self.p = len(self.As) # number of operators
        self.ns = [] # list of shapes
        self.idxs = []
        tmp = 0
        for j, op in enumerate(self.As):
            nj = op.shape[0]
            assert op.shape[1] == nj, "Not all A_i are square!"
            self.ns.append(nj)

            if j == 0:
                self.idxs.append( np.arange(nj) )     
            else:
                self.idxs.append( np.arange(tmp, tmp+nj) )
                
            tmp += nj


        # Set shape
        n = sum(self.ns)
        shape = (n, n)

        # Define matvecs
        def _matvec(x):
            pieces = []
            for j, op in enumerate(self.As):
                pieces.append( op.matvec(x[self.idxs[j]]) )
            return np.hstack(pieces)
        
        def _rmatvec(x):
            pieces = []
            for j, op in enumerate(self.As):
                pieces.append( op.rmatvec(x[self.idxs[j]]) )
            return np.hstack(pieces)

            
        super().__init__(shape, _matvec, _rmatvec, device=device)
                
                
        
    def to_gpu(self):

        As_gpu = []
        for op in self.As:
            gpu_op = op.to_gpu()
            As_gpu.append(gpu_op)

        return BlockDiagonalOperator(As_gpu, device="gpu")
    
    def to_cpu(self):

        As_cpu = []
        for op in self.As:
            cpu_op = op.to_cpu()
            As_cpu.append(cpu_op)

        return BlockDiagonalOperator(As_cpu, device="cpu")