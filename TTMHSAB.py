
import numpy as np
from numba import njit
import time
from utils import signal_func, signal_grad
from utils import new_nonmonotone,new_nonmonotone_ttmhsab

@njit
def TTMHSAB(n, A, b, lamda,psi, 
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
    Output_n2re = [np.linalg.norm(x0 - xs)/np.linalg.norm(xs)]

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
    while norm_gk >= Tolerance * (1+abs(fk)) and NI < MaxIter:
        # Initialize parameters
        sigma = 0.75
        delta = 1e-4
        lambda_ = 0.85
        M = 10
        alpha0 = 1.0
        # During optimization loop
        alpha, fk1, LSFval, mem_count = new_nonmonotone_ttmhsab(
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
        yk = gk1 - gk
        # BEGIN SEARCH DIRECTION update
        # Compute vk value!
        vk = np.linalg.norm(sk)/np.linalg.norm(yk)

        if np.isnan(vk) or np.isinf(vk):
            vk = np.dot(sk, sk)
        denum=np.dot(sk, yk)
        if np.isnan(denum) or np.isinf(denum):
            denum = np.dot(sk, sk)

        dkii = vk - 2*vk*(sk*yk)/(denum)+(1+vk*(np.dot(yk,yk)/denum))*(sk**2)/denum
        EPSL = 1e-10
        if denum != 0: 
            DK= np.minimum(np.maximum(dkii, EPSL), EPSL**(-1))
        else:
            DK = np.ones(len(x0))

        yk_bar = DK**(-1)*sk # Element-wise division

        denum = np.dot(dk, yk_bar)

        # Compute betak and thetak
        betak = np.dot(gk1, yk_bar) / denum  # NMHS
        thetak = np.dot(gk1, dk) / denum  # NM-Thetak

        # Update search direction
        dk = -gk1 + betak*dk - thetak*yk_bar
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
