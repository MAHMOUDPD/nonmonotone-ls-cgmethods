'''
Written by: Mahmoud M. Yahaya
Contact address: mahmoudpd@gmail.com
Date of last update: 06/08/2025
'''
import numpy as np
from numba import njit
import time
from utils import signal_func, signal_grad
from utils import new_nonmonotone

@njit
def NTDCG1(n, A, b, lamda,psi, 
               init, xs, MaxIter, NLmax, CRestart, Tolerance, delta, sigma):
            #   options):
    # if options['init'] == 0:
    #     x = np.zeros(n)
    # elif options['init'] == 1:
    #     x = np.linalg.norm(A.T @ b, np.inf) * np.ones(n)
    # elif options['init'] == 2:
    #     x = A.T @ b
    # elif options['init'] == 3:
    #     x = np.random.randn(n)
    # elif options['init'] == 4:
    #     x = 2 * (np.random.rand(n) - 0.5)

    # xs = options['xs']
    if init == 0:
        x = np.zeros(n)
    elif init == 1:
        x = np.linalg.norm(A.T @ b, np.inf) * np.ones(n)
    elif init == 2:
        x = A.T @ b
    elif init == 3:
        x = np.random.randn(n)
    elif init == 4:
        x = 2 * (np.random.rand(n) - 0.5)

    x0 = x.copy()
    Output_f = []
    Output_n2re = [np.linalg.norm(x0 - xs) / np.linalg.norm(xs)]

    Nf = 1
    Ng = 1
    NI = 0

    # func = lambda x: signal_func(x, lamda, A, b, psi)
    # gfunc = lambda x: signal_grad(x, lamda, A, b, psi)
    
    fk = signal_func(x0, lamda, A, b, psi)
    gk = signal_grad(x0, lamda, A, b, psi)
    dk = -gk
    gkdk = np.dot(gk, dk)
    norm_gk = np.linalg.norm(gk)
    b0 = np.ones(len(x0))
    b1= np.zeros(len(x0))
    tk_try = 1.0 / norm_gk

    # start_time = time.time()
        # Pre-allocate memory buffer
    M =10
    memory_arr = np.zeros(M)
    mem_count = 0
    while norm_gk >= Tolerance * (1 + abs(fk)) and NI < MaxIter:
        # Initialize parameters
        sigma = 0.75
        delta = 1e-3
        lambda_ = 0.85
        M =10
        alpha0 = 1.0
        # During optimization loop
        alpha, fk1, LSFval, mem_count = new_nonmonotone(
            myfunc3=signal_func,
            fk=fk,
            x0=x0,
            lamda=lamda,
            A=A,
            b=b,
            psi=psi,
            dk=dk,
            gk=gk,
            gkdk=gkdk,
            alpha0=alpha0,
            sigma=sigma,
            delta=delta,
            lambda_=lambda_,
            M=M,
            memory_arr=memory_arr,
            mem_count=mem_count
        )
        # gk1dk = np.dot(gk1, dk)

        sk = alpha * dk
        x1 = x0 + sk
        gk1 = signal_grad(x1, lamda, A, b, psi) 
        Ng += 1
        norm_gk1 = np.linalg.norm(gk1)
        yk = gk1 -gk
        # BEGIN SEARCH DIRECTION update
        vk = np.dot(sk, yk)
        if np.isnan(vk) or np.isinf(vk):
            vk = np.dot(sk, sk)

        MU = 1e-8
        EPSL = 1e-10

        # Compute BB step sizes
        alph1 = np.dot(sk, sk) / vk  # BB1
        alph2 = vk / np.dot(yk, yk)  # BB2

        # Compute internal matrix with element-wise operations
        numerator = sk * yk + MU * b0
        denominator = sk**2 + MU
        internal_matrix = numerator / denominator

        # Clip values to [alph1, alph2] range
        # b1 = np.copy(internal_matrix)
        b1[internal_matrix < alph1] = alph1
        b1[internal_matrix > alph2] = alph2
        # OR USE!!
        # b1 = np.clip(internal_matrix, alph2, alph1)

        # Compute zk and denominator term
        zk = sk / b1  # Element-wise division
        denum = np.dot(dk, zk)

        # Compute betak and thetak
        betak = np.dot(gk1, zk) / denum  # NMHS
        thetak = np.dot(gk1, dk) / denum  # NM-Thetak

        # Update search direction
        dk = -gk1 + betak * dk - thetak * zk
        # END OF DIRECTION!
        if np.dot(dk, gk1) > -CRestart*np.linalg.norm(dk)*np.linalg.norm(gk1):
            dk = -gk1

        NI += 1
        gk1dk1 = np.dot(gk1, dk)
        tk_try *= gkdk / gk1dk1

        x0 = x1
        b0 = b1
        gk = gk1
        norm_gk = norm_gk1
        fk = fk1
        gkdk = gk1dk1

        Output_f.append(fk1)
        Output_n2re.append(np.linalg.norm(x0-xs)/np.linalg.norm(xs))
    # elapsed_time = time() - start_time
    xopt = x1
    # fopt = fk
    gopt = gk
    TNF = Nf+3*Ng
    nz_x = np.count_nonzero(xopt)

    mse = np.linalg.norm(xopt-xs)/np.linalg.norm(xs)

    return xopt, Output_f[:NI], gopt, NI, Nf, Ng, TNF, mse, nz_x,  Output_n2re[:NI+1]
