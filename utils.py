'''
Written by: Mahmoud M. Yahaya
Contact address: mahmoudpd@gmail.com
Date of last update: 06/08/2025
'''

import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import hadamard, qr
from scipy.sparse import diags
from numba import njit

# Core functions compatible with @njit ----------------------------------------

@njit
def new_nonmonotone(myfunc3, fk, x0,lamda,A,b,psi,dk,gk,gkdk,alpha0,sigma,delta,lambda_,M,memory_arr,mem_count):
    """
    Nonmonotone line search compatible with Numba JIT    
    Parameters:
    myfunc3 : njit-compatible objective function
    fk : current function value
    x0 : current point
    dk : search direction
    gkdk : directional derivative (gradient dot direction)
    alpha0 : initial step size
    sigma : reduction factor (0 < sigma < 1)
    delta : Armijo condition constant
    lambda_ : forgetting factor (0 < lambda_ < 1)
    M : memory length
    memory_arr : pre-allocated memory buffer (size M)
    mem_count : current elements in memory buffer
    
    Returns:
    alpha : computed step size
    f_next : function value at new point
    LSFval : function evaluations count
    updated_mem_count : updated memory element count
    """
    # Update memory buffer using circular indexing
    if M > 0:
        if mem_count < M:
            memory_arr[mem_count] = fk
            mem_count += 1
        else:
            # Shift elements to maintain sliding window
            for i in range(M-1):
                memory_arr[i] = memory_arr[i+1]
            memory_arr[M-1] = fk
    # Compute weighted maximum fbar
    if mem_count > 0:
        fbar = -np.inf
        for i in range(mem_count):
            weight = lambda_ ** i
            weighted_val = memory_arr[i] * weight
            if weighted_val > fbar:
                fbar = weighted_val
    else:
        fbar = fk  # Fallback to current value if no history
    # Compute reference value Rk
    etak = 0.85
    Rk = etak * fbar + (1 - etak) * fk
    # Backtracking line search
    alpha = alpha0
    it = 0
    LSFval = 0
    while True:
        x1 = x0 + alpha * dk
        f_next = myfunc3(x1, lamda, A, b, psi)
        LSFval += 1
        # Check termination conditions
        if f_next <= Rk + delta * alpha * gkdk or it >= 50:
            break
        # Reduce step size
        alpha *= sigma
        it += 1
    return alpha, f_next, LSFval, mem_count

@njit
def new_nonmonotone_ttmhsab(myfunc3, fk, x0,lamda,A,b,psi,dk,gk,gkdk,alpha0,sigma,delta,lambda_,M,memory_arr,mem_count):
    """
    New_nonmonotone_ttmhsab line search compatible with Numba JIT    
    Parameters:
    myfunc3 : njit-compatible objective function
    fk : current function value
    x0 : current point
    dk : search direction
    gkdk : directional derivative (gradient dot direction)
    alpha0 : initial step size
    sigma : reduction factor (0 < sigma < 1)
    delta : Armijo condition constant
    lambda_ : forgetting factor (0 < lambda_ < 1)
    M : memory length
    memory_arr : pre-allocated memory buffer (size M)
    mem_count : current elements in memory buffer
    
    Returns:
    alpha : computed step size
    f_next : function value at new point
    LSFval : function evaluations count
    updated_mem_count : updated memory element count
    """
    # Update memory buffer using circular indexing
    if M > 0:
        if mem_count < M:
            memory_arr[mem_count] = fk
            mem_count += 1
        else:
            # Shift elements to maintain sliding window
            for i in range(M-1):
                memory_arr[i] = memory_arr[i+1]
            memory_arr[M-1] = fk
    # Compute weighted maximum fbar
    if mem_count > 0:
        fbar = -np.inf
        for i in range(mem_count):
            weight = lambda_ ** i
            weighted_val = memory_arr[i] * weight
            if weighted_val > fbar:
                fbar = weighted_val
    else:
        fbar = fk  # Fallback to current value if no history
    # Compute reference value Rk
    Rk = fbar

    # etak = 0.85
    # Rk = etak * fbar + (1 - etak) * fk

    # Backtracking line search
    alpha = alpha0
    it = 0
    LSFval = 0
    while True:
        x1 = x0 + alpha * dk
        f_next = myfunc3(x1, lamda, A, b, psi)
        LSFval += 1
        # Check termination conditions
        if f_next <= Rk + delta * alpha * gkdk or it >= 50:
            break
        # Reduce step size
        alpha *= sigma
        it += 1
    return alpha, f_next, LSFval, mem_count



# def new_nonmonotone(myfunc3, fk, x0, dk, gkdk, gk, alpha0, params):
#     # Set default parameters if not provided
#     if 'sigma' not in params:
#         params['sigma'] = 0.5
#     if 'delta' not in params:
#         params['delta'] = 1e-4
#     if 'lambda' not in params:
#         params['lambda'] = 0.85
#     if 'M' not in params:
#         params['M'] = 10  # Initialize M as None if not provided

#     # Extract parameters
#     sigma = params['sigma']
#     delta = params['delta']
#     lambda_ = params['lambda']  # Use lambda_ to avoid keyword conflict
#     M = params['M']

#     # Initialize variables
#     alpha = alpha0
#     memory = params.get('memory', [])  # Retrieve memory from params or initialize empty
    
#     # Update memory with current function value
#     memory.append(fk)
#     if M is not None and len(memory) > M:
#         memory = memory[-M:]  # Keep only the last M elements
    
#     # Compute weights for nonmonotone condition
#     weights = np.array([lambda_ ** i for i in range(len(memory))])
#     weighted_memory = weights * np.array(memory)
#     fbar = np.max(weighted_memory)
    
#     # Compute etak (using fixed value 0.85 as in MATLAB code)
#     etak = 0.85
#     Rk = etak * fbar + (1 - etak) * fk

#     # Backtracking line search
#     it = 0
#     LSFval = 0  # Count function evaluations
#     while True:
#         x1 = x0 + alpha * dk
#         f_next = myfunc3(x1)
#         LSFval += 1
        
#         # Check nonmonotone Armijo condition or iteration limit
#         if f_next <= Rk + delta * alpha * gkdk or it >= 50:
#             break
        
#         # Reduce step size
#         alpha *= sigma
#         it += 1
    
#     # Update memory in params for next iteration
#     params['memory'] = memory
    
#     return alpha, f_next, LSFval

@njit
def signal_func(x, lamda, A, b, psi):
    """Objective function with smoothed L1 regularization"""
    f_val = x * np.tanh(x / lamda)
    residual = A @ x - b
    F = 0.5 * np.sum(residual**2) + psi * np.sum(f_val)
    return F

@njit
def signal_grad(x, lamda, A, b, psi):
    """Gradient of the objective function"""
    dg_val = (x / lamda) * (1 - np.tanh(x / lamda)**2) + np.tanh(x / lamda)
    JF = A.T @ (A @ x - b) + psi * dg_val
    return JF

@njit
def pfft(x, mode, n, picks):
    """Partial Fourier transform operator"""
    if mode == -1:
        return picks
    elif mode == 1:
        y = fft(x) / np.sqrt(n)
        return y[picks]
    elif mode == 2:
        y = np.zeros(n, dtype=np.complex128)
        y[picks] = x
        return ifft(y) * np.sqrt(n)
    else:
        raise ValueError("mode must be 1 (for A*x) or 2 (for A'*x)")

# Helper functions (not njit compatible) -------------------------------------

def getM_mu(full, m, n, Ameth, A, b, sig1, sig2, alpha):
    """
    Calculates M and mu for problem scaling.
    Note: Simplified version without chi2inv dependency
    """
    # Simplified implementation - for demonstration only
    if Ameth in {2, 6}:  # Orthonormal rows
        M = None
        mu = 1.0
        sig = np.sqrt(sig1**2 + sig2**2)
    elif Ameth in {4, 5}:  # Partial Hadamard/Fourier
        M = None
        mu = 1.0
        sig = np.sqrt(sig1**2 + sig2**2 / n)
    else:  # General case
        M = None
        smax = np.linalg.norm(A, 2)
        A = A / smax
        b = b / smax
        mu = 1.0
        sig = np.sqrt(sig1**2 + sig2**2)
    
    return M, mu, A, b, sig, None, None, None

def getData(m, n, k, Ameth, xmeth, sigma1=0, sigma2=0, state=None):
    """Generates sample compressed sensing problems"""
    if state is not None:
        np.random.seed(state)
    
    # Generate matrix A
    if Ameth == 0:
        A = np.random.randn(m, n)
    elif Ameth == 1:
        A = np.random.randn(m, n)
        norms = np.linalg.norm(A, axis=0)
        A = A @ diags(1/norms)
    elif Ameth == 2:
        A = np.random.randn(n, m)
        A, _ = qr(A, mode='economic')
        A = A.T
    elif Ameth == 3:
        A = np.sign(np.random.rand(m, n) - 0.5)
        A[A == 0] = 1
    elif Ameth == 4:
        H = hadamard(n)
        picks = np.random.choice(n, m, replace=False)
        A = H[picks, :]
    elif Ameth == 5:
        picks = np.random.choice(n, m, replace=False)
        A = lambda x, mode: pfft(x, mode, n, picks)
    elif Ameth == 6:
        raise NotImplementedError("DCT not implemented in this version")
    
    # Generate sparse signal xs
    xs = np.zeros(n)
    p = np.random.choice(n, k, replace=False)
    
    if xmeth == 0:
        xs[p] = 2 * np.random.randn(k)
    elif xmeth == 1:
        xs[p] = 2 * (np.random.rand(k) - 0.5)
    elif xmeth == 2:
        xs[p] = np.ones(k)
    elif xmeth == 3:
        xs[p] = np.sign(np.random.randn(k))
    elif xmeth == 4:
        xs[p] = np.where(np.random.rand(k) < 0.5, 
                         10**(4*np.random.rand(k)), 
                         -10**(4*np.random.rand(k)))
    
    # Add noise to signal
    xsn = xs + sigma1 * np.random.randn(n)
    
    # Generate measurements
    if callable(A):
        b = A(xsn, 1)
    else:
        b = A @ xsn
    
    # Add measurement noise
    b += sigma2 * np.random.randn(m)
    
    return A, b, xs, xsn
