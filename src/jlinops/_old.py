
















# class BaseLinearOperator(LinearOperator):
#     """Base class for representing a linear operators. Built as a subclass of SciPy's LinearOperator.
#     """
    
#     def __init__(self, dtype, shape, device="cpu"):

#         # Device for cpu/gpu
#         valid_devices = ["cpu", "gpu"]
#         assert device in valid_devices, f"device must be one of {valid_devices}."
#         self.device = device

#         super().__init__(dtype, shape)

#     def _matvec(self, x):

#         raise NotImplementedError



# class CustomLinearOperator(BaseLinearOperator):
#     """Class for implementing linear operators from matvec/rmatvec functions.
#     """

#     """Linear operator defined in terms of user-specified operations."""

#     def __init__(self, shape, matvec, rmatvec=None, matmat=None,
#                  dtype=None, rmatmat=None, device="cpu"):
        
#         super().__init__(dtype, shape, device=device)

#         self.args = ()

#         self.__matvec_impl = matvec
#         self.__rmatvec_impl = rmatvec
#         self.__rmatmat_impl = rmatmat
#         self.__matmat_impl = matmat

#         self._init_dtype()

#     def _matmat(self, X):
#         if self.__matmat_impl is not None:
#             return self.__matmat_impl(X)
#         else:
#             return super()._matmat(X)

#     def _matvec(self, x):
#         return self.__matvec_impl(x)

#     def _rmatvec(self, x):
#         func = self.__rmatvec_impl
#         if func is None:
#             raise NotImplementedError("rmatvec is not defined")
#         return self.__rmatvec_impl(x)

#     def _rmatmat(self, X):
#         if self.__rmatmat_impl is not None:
#             return self.__rmatmat_impl(X)
#         else:
#             return super()._rmatmat(X)

#     def _adjoint(self):
#         return CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
#                                      matvec=self.__rmatvec_impl,
#                                      rmatvec=self.__matvec_impl,
#                                      matmat=self.__rmatmat_impl,
#                                      rmatmat=self.__matmat_impl,
#                                      dtype=self.dtype)

