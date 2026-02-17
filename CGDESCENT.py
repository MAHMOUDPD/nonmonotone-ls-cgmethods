import numpy as np
from numba import njit
import time
from utils import signal_func, signal_grad


@njit
def CGDESCENT(n, A, b, lamda,psi, 
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

    # xs = options['xs']
    x0 = x.copy()

    Output_f = []
    Output_n2re = [np.linalg.norm(x0 - xs) / np.linalg.norm(xs)]

    # MaxIter = options['MaxIter']
    # NLmax = options['NLmax']
    # CRestart = options['CRestart']
    # Tolerance = options['Tolerance']
    # delta = options['delta']
    # sigma = options['sigma']

    Nf = 1
    Ng = 1
    NI = 0

    fk = signal_func(x0, lamda, A, b, psi)
    gk = signal_grad(x0, lamda, A, b, psi)

    dk = -gk
    gkdk = np.dot(gk, dk)
    norm_gk = np.linalg.norm(gk)
    tk_try = 1.0 / norm_gk

    # start = time.time()

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
        while NL <= NLmax:
            if fk1 > fk + delta * tk_try * gkdk:
                tk_up = tk_try
            elif gk1dk < sigma * gkdk:
                tk_low = tk_try
            else:
                break

            if tk_up == np.inf:
                tk_try = 2 * tk_try
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
        numgk = np.linalg.norm(gk)
        numdk = np.linalg.norm(dk)

        eta_const = 0.01
        yk = gk1 - gk
        etak = -1 / (numdk * min(eta_const, numgk))
        numk = np.dot(dk, yk)

        betakN = (np.dot(yk, gk1) / numk - (2 * np.dot(gk1, dk) * np.dot(yk, yk)) / (numk**2))
        barbetakN = max(betakN, etak)

        dk = -gk1 + barbetakN * dk

        if np.dot(dk, gk1) > -CRestart * np.linalg.norm(dk) * np.linalg.norm(gk1):
            dk = -gk1

        NI += 1

        gk1dk1 = np.dot(gk1, dk)
        tk_try = tk_try * gkdk / gk1dk1

        x0 = x1.copy()
        gk = gk1.copy()
        norm_gk = norm_gk1
        fk = fk1
        gkdk = gk1dk1

        Output_f.append(fk1)
        Output_n2re.append(np.linalg.norm(x0 - xs) / np.linalg.norm(xs))

    # elapsed = time.time() - start
    xopt = x1
    # fopt = fk
    gopt = gk
    TNF = Nf + 3 * Ng
    nz_x = np.count_nonzero(xopt)

    mse = np.linalg.norm(xopt - xs) / np.linalg.norm(xs)


    return xopt, Output_f[:NI], gopt, NI, Nf, Ng, TNF, mse, nz_x,  Output_n2re[:NI+1]


    # return {
    #     'x': xopt,
    #     'f': np.array(Output_f),
    #     'n2re': np.array(Output_n2re),
    #     'cpu': -1,
    #     'iterfinal': NI,
    #     'nf': Nf,
    #     'ng': Ng,
    #     'tnf': TNF,
    #     'mse': np.linalg.norm(xopt - xs) / np.linalg.norm(xs),
    #     'nz': nz_x
    # }
