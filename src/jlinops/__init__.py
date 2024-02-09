
# Module variables
CUPY_INSTALLED = False
try:
    import cupy
    CUPY_INSTALLED = True
except:
    pass


# Imports
from .util import *
from .base import *
from .base import _CustomLinearOperator, _AdjointLinearOperator
from .linalg import *
from .diagonal import *
from .stacked import *
from .blurring import *
from .local_averaging import *
from .subsampling import *
from .local_averaging import *
from .derivatives import *
from .wavelets import *
from .structured import *
from .interpolation import *
from .linear_solvers import *
from .inv import *
from .cholesky import *
from .cginv import *
from .pseudoinverse import *
from .oblique import *
from .data import *
from .proximal import *

from .linalg import *





# Imports
# from .matrix import *
# from .pseudoinverse import *
# from .cginv import *
# from .blurring import *
# from .subsampling import *
# from .diagonalized import *
# from .diagonal import *
# from .blurring import *
# from .util import *
# from .derivatives import *
# from .identity import *
# from .linear_solvers import cg

# from .base import *




# from .matrix import MatrixOperator



# __matrix__ = [
#     "MatrixOperator",
# ]

# __all__ = __matrix__

















