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
            x = x.reshape(-1)
            pieces = []
            for j, op in enumerate(self.As):
                pieces.append( op.matvec(x[self.idxs[j]]) )
            return np.hstack(pieces)
        
        def _rmatvec(x):
            x = x.reshape(-1)
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
    






class RectBlockDiagonalOperator(_CustomLinearOperator):
    """Given (rectangular) operators A_1, \ldots, A_p of shapes m_i \times n, represents the block diagonal operator diag(A_1, \ldots, A_p).
    """
    def __init__(self, As, device="cpu", type=1):

        # Setup
        self.As = As # list of operators
        self.p = len(self.As) # number of operators
        self.n = As[0].shape[1]
        self.ms = [] # list of shapes
        self.idxs = []
        self.mvec_idxs = []
        self.rmvec_idxs = []
        mvec_tmp = 0
        rmvec_tmp = 0
        for j, op in enumerate(self.As):
            mj = op.shape[0]
            assert op.shape[1] == self.n, "Not all A_i have same number of columns!"
            self.ms.append(mj)

            if j == 0:
                self.mvec_idxs.append( np.arange(self.n) )
                self.rmvec_idxs.append( np.arange(mj) )
                #self.idxs.append( np.arange(nj) )     
            else:
                self.mvec_idxs.append( np.arange(mvec_tmp, mvec_tmp+self.n) )
                self.rmvec_idxs.append( np.arange(rmvec_tmp, rmvec_tmp+mj) )
                #self.idxs.append( np.arange(tmp, tmp+nj) )
                
            mvec_tmp += self.n
            rmvec_tmp += mj


        # Set shape
        n_tot = len(As)*self.n
        shape = (sum(self.ms), n_tot)
        #print(shape)

        # Define matvecs
        def _matvec(x):
            x = x.reshape(-1)
            pieces = []
            for j, op in enumerate(self.As):
                pieces.append( op.matvec(x[self.mvec_idxs[j]]) )
            return np.hstack(pieces)
        
        def _rmatvec(x):
            x = x.reshape(-1)
            pieces = []
            for j, op in enumerate(self.As):
                pieces.append( op.rmatvec(x[self.rmvec_idxs[j]]) )
            return np.hstack(pieces)

            
        super().__init__(shape, _matvec, _rmatvec, device=device)
                
                
        
    def to_gpu(self):

        As_gpu = []
        for op in self.As:
            gpu_op = op.to_gpu()
            As_gpu.append(gpu_op)

        return RectBlockDiagonalOperator(As_gpu, device="gpu")
    
    def to_cpu(self):

        As_cpu = []
        for op in self.As:
            cpu_op = op.to_cpu()
            As_cpu.append(cpu_op)

        return RectBlockDiagonalOperator(As_cpu, device="cpu")