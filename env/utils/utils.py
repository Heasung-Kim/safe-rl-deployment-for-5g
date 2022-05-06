import numpy as np
from numpy.linalg import inv, norm

def array_to_complex_predocer(common_precoder):
    """
    This function is to get normalized common message matrix with common - precoder ( (8,) vector ) which is
    calculating, and optimized by CMA-ES.

    :param common_precoder: Last 8 components in action vector in cma_precoder.py
    :return w_c : normalized common message vector (4,) vector
    """
    common_precoder = np.array(common_precoder)
    w_c_real = common_precoder[:int(len(common_precoder)/2)]  # Previous 4 components in common-precoder
    w_c_img = common_precoder[int(len(common_precoder)/2):]  # Last 4 components in common-precoder
    w_c = w_c_real + 1j * w_c_img
    w_c_size = norm(w_c, 2)  # vector
    w_c = w_c / (w_c_size + 0.000000001)

    return w_c