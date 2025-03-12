import numpy as np
from numba import njit


@njit
def m4_numba(xI, xJ, h):
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    q = r / h
    A = 40.0 / (h**2 * 7 * np.pi)

    if q > 1:
        return 0.0
    elif q > 0.5:
        return A * (2 * (1 - q) ** 3)
    else:
        return A * (6 * (q**3 - q**2) + 1)


@njit
def grad_m4_numba(xI, xJ, h):
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)

    # Avoid division by zero:
    if r < 1e-5:
        return np.array([0.0, 0.0])

    q = r / h
    A = 40.0 / (7 * np.pi * h**2)
    dW_dq = 0.0

    if q > 1:
        dW_dq = 0.0
    elif q > 0.5:
        dW_dq = -6.0 * A * (1 - q) ** 2
    else:
        dW_dq = A * (18.0 * q**2 - 12.0 * q)

    # Chain rule: dW/dr = dW/dq * (1/h)
    dW_dr = dW_dq / h

    # Return gradient: dW/dr * ( (xI - xJ)/r )
    return dW_dr * np.array([dx, dy]) / r


@njit
def w(xI, xJ, h):
    return m4_numba(xI, xJ, h)


@njit
def grad_w(xI, xJ, h):
    return grad_m4_numba(xI, xJ, h)
