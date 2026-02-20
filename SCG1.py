'''
Written by: Mahmoud M. Yahaya
Contact address: mahmoudpd@gmail.com
Date of last update: 06/08/2025
'''

import numpy as np
from numba import njit
import time
from utils import signal_func, signal_grad


@njit
def SCG1(n, A, b, lamda,psi, 
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

    fk = signal_func(x0, lamda, A, b, psi)
    gk = signal_grad(x0, lamda, A, b, psi)

    dk = -gk
    gkdk = np.dot(gk, dk)
    norm_gk = np.linalg.norm(gk)

    tk_try = 1.0 / norm_gk

    # start_time = time.time()

    while norm_gk >= Tolerance * (1 + abs(fk)) and NI < MaxIter:
        tk_low = 0.0
        tk_up = np.inf

        x1 = x0 + tk_try * dk
        fk1 = signal_func(x1, lamda, A, b, psi)
        Nf += 1
        gk1 = signal_grad(x1, lamda, A, b, psi)
        Ng += 1
        gk1dk = np.dot(gk1, dk)
        NL = 1
        sigma1 = 0.1

        while NL <= NLmax:
            if fk1 > fk - delta * tk_try * np.dot(dk, dk):
                tk_up = tk_try
            elif gk1dk < sigma1 * gkdk:
                tk_low = tk_try
            else:
                break
            if tk_up == np.inf:
                tk_try *= 2.0
            else:
                tk_try = 0.5 * (tk_low + tk_up)

            x1 = x0 + tk_try * dk
            fk1 = signal_func(x1, lamda, A, b, psi)
            Nf += 1
            gk1 = signal_grad(x1, lamda, A, b, psi)
            Ng += 1
            gk1dk = np.dot(gk1, dk)
            NL += 1

        sk = tk_try * dk
        x1 = x0 + sk
        norm_gk1 = np.linalg.norm(gk1)

        # SCG1 direction update
        yk = gk1 - gk
        rho1_ =0.01
        rho2_ = rho1_
        normd = np.linalg.norm(dk)
        thetak1=rho1_+np.linalg.norm(yk)/normd
        thetak2=rho2_+norm_gk1/normd
        
        if norm_gk1**2 > np.abs(np.dot(gk1, yk)):
            thetak = thetak1
        else:
            thetak=thetak2
        betak = np.dot(gk1,yk)/(normd**2)+np.dot(gk1,dk)/(normd**2)
        dk = -thetak*gk1+betak*dk

        if np.dot(dk, gk1) > -CRestart * np.linalg.norm(dk) * np.linalg.norm(gk1):
            dk = -gk1

        NI += 1
        gk1dk1 = np.dot(gk1, dk)
        tk_try *= gkdk / gk1dk1

        x0 = x1
        gk = gk1
        norm_gk = norm_gk1
        fk = fk1
        gkdk = gk1dk1

        Output_f.append(fk1)
        Output_n2re.append(np.linalg.norm(x0 - xs) / np.linalg.norm(xs))

    # elapsed_time = time() - start_time

    xopt = x1
    # fopt = fk
    gopt = gk
    TNF = Nf + 3 * Ng
    nz_x = np.count_nonzero(xopt)

    mse = np.linalg.norm(xopt - xs) / np.linalg.norm(xs)


    return xopt, Output_f[:NI], gopt, NI, Nf, Ng, TNF, mse, nz_x,  Output_n2re[:NI+1]
