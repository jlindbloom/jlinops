# import numpy as np
# import matplotlib.pyplot as plt

# from .derivative_matrices import build_1d_first_order_derivative



# def build_explicit_anisotropic_derivative(input_shape, order=1, boundary="periodic"):
#     """
#     Builds a LinearOperator for the anisotropic discrete gradient.
#     """

#     # Handle inputs

#     valid_orders = [1]
#     assert order in valid_orders, f"Invalid order; must be in {valid_orders}."

#     valid_boundaries = ["none", "periodic", "zero"]
#     assert boundary in valid_boundaries, f"Invalid boundary; must be in {valid_boundaries}."

#     n_dim = len(input_shape)
#     assert (n_dim == 1), "Dimension must be one."
#     #assert (n_dim == 1) or (n_dim == 2), "Dimension must be either one or two."

#     # Redirect to appropriate functions

#     if n_dim == 1:

#         if order == 1:
#             N = input_shape[0]
#             diff_mat = build_1d_first_order_derivative(N, boundary=boundary)
#             return MatrixOperator(diff_mat)
#         # elif order == 2:
#         #     N = input_shape[0]
#         #     diff_mat = build_diff_mat_2nd_derivative(N, boundary=boundary)
#         #     return MatrixOperator(diff_mat)
#         else:
#             raise NotImplementedError

#     # elif n_dim == 2:

#     #     if order == 1:
#     #         M, N = input_shape
#     #         diff_mat = build_2d_first_order_grad(M, N, boundary=boundary)
#     #         return MatrixOperator(diff_mat, input_shape=(M,N), output_shape=(diff_mat.shape[0],))
#     #     elif order == 2:
#     #         M, N = input_shape
#     #         diff_mat = build_diff_mat_2nd_order_2d(M, N, boundary=boundary)
#     #         return MatrixOperator(diff_mat, input_shape=(M,N), output_shape=(diff_mat.shape[0],))
#     #     else:
#     #         raise NotImplementedError

#     else:
#         raise NotImplementedError


