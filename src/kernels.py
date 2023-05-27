import numpy as np


# linear kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# infinite polynomial kernel
def inf_poly_kernel(x1, x2):
    return 1/(1-np.dot(x1, x2))