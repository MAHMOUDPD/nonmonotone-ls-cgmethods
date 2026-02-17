import numpy as np
from numba import njit

@njit
def signal_grad(x, lamda, A, b, psi):
    # Compute dg_val element-wise using vectorized operations
    dg_val = (x / lamda) * (1 - np.tanh(x / lamda)**2) + np.tanh(x / lamda)
    
    # Compute the gradient: A^T @ (A @ x - b) + psi * dg_val
    JF = A.T @ (A @ x - b) + psi * dg_val
    return JF