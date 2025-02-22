import numpy as np
from numba import njit

# Normalization factors for the cubic spline 0-1 kernel in 2D
CUBIC_SPLINE_01_FACTOR = 40.0 / (7 * np.pi)  # 2D normalization factor
CUBIC_SPLINE_01_GRADIENT_FACTOR = CUBIC_SPLINE_01_FACTOR * 6.0  # Gradient factor
__CUBIC_KERNEL_FACTOR = 10 / (7 * np.pi
                              )  # Normalization for cubic spline kernel


@njit
def cubic_kernel_numba(xI, xJ, h):
    """
    Computes the cubic spline smoothing kernel value in 2D.
    
    Uses the piecewise function:
        - If 0 <= q <= 1:  W = (1 - 1.5*q^2 + 0.75*q^3)
        - If 1 < q < 2:    W = 0.25 * (2 - q)^3
        - If q >= 2:       W = 0
    
    :param xI: Position of particle I (NumPy array)
    :param xJ: Position of particle J (NumPy array)
    :param h: Support radius
    :return: Kernel value W
    """
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)

    half_h = h / 2.0
    k = __CUBIC_KERNEL_FACTOR / (half_h**2)
    q = r / half_h

    if q >= 2.0:
        return 0.0
    elif q <= 1.0:
        q2 = q * q
        return k * (1 - 1.5 * q2 + 0.75 * q2 * q)
    else:
        return k * 0.25 * (2 - q)**3


@njit
def cubic_grad_kernel_numba(xI, xJ, h):
    """
    Computes the gradient of the cubic spline smoothing kernel in 2D.

    Uses the piecewise function:
        - If 0 <= q <= 1:  grad_W = (-3*q + 2.25*q^2) * (r / |r|)
        - If 1 < q < 2:    grad_W = -0.75 * (2 - q)^2 * (r / |r|)
        - If q >= 2:       grad_W = 0

    :param xI: Position of particle I (NumPy array)
    :param xJ: Position of particle J (NumPy array)
    :param h: Support radius
    :return: Gradient of the cubic spline kernel (NumPy array)
    """
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)

    if r < 1e-5:
        return np.array([0.0, 0.0])  # Avoid division by zero

    half_h = h / 2.0
    k = __CUBIC_KERNEL_FACTOR / (half_h**2)
    q = r / half_h
    inv_r = 1.0 / r if r > 1e-5 else 0.0  # Safe inverse

    grad_w = 0.0
    if q < 1.0:
        grad_w = (k / half_h) * (-3.0 * q + 2.25 * q * q)
    elif q < 2.0:
        grad_w = -0.75 * (k / half_h) * (2.0 - q)**2

    return grad_w * inv_r * np.array([dx, dy])


@njit
def cubic_kernel_01_numba(xI, xJ, h):
    """
    Computes the cubic spline kernel (0-1 variant) in 2D.
    Based on the kernel presented in:
    Dan Koschier, Jan Bender, Barbara Solenthaler, & Teschner, M. (2019). 
    Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids. 
    Eurographics Tutorial 2019.

    :param xI: Position of particle I
    :param xJ: Position of particle J
    :param h: Smoothing length
    :return: Kernel value
    """
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    q = r / h if r > 1e-5 else 0.0

    if q >= 1.0:
        return 0.0
    elif q <= 0.5:
        return CUBIC_SPLINE_01_FACTOR * (6 * q**3 - 6 * q**2 + 1)
    else:
        return 2 * CUBIC_SPLINE_01_FACTOR * (1 - q)**3


@njit
def cubic_grad_kernel_01_numba(xI, xJ, h):
    """
    Computes the gradient of the cubic spline kernel (0-1 variant).

    :param xI: Position of particle I
    :param xJ: Position of particle J
    :param h: Smoothing length
    :return: Gradient (NumPy array)
    """
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)

    if r < 1e-5:
        return np.array([0.0, 0.0])

    q = r / h
    dWdr = 0.0
    if q < 1.0:
        if q <= 0.5:
            dWdr = CUBIC_SPLINE_01_GRADIENT_FACTOR / h * (3 * q**2 - 2 * q)
        else:
            dWdr = -CUBIC_SPLINE_01_GRADIENT_FACTOR / h * (1 - q)**2

    return (dWdr / r) * np.array([dx, dy])


@njit
def m4_numba(xI, xJ, h):
    """
    Computes the additional kernel value in 2D:
    
        W(x, h) = 1/h^2 * 
            { (15/(14π))*[(2 - q)^3 - 4*(1 - q)^3],  for 0 <= q <= 1,
              (15/(14π))*(2 - q)^3,                  for 1 <= q <= 2,
              0,                                    for q > 2 }
              
    where q = |xI - xJ| / h.
    
    :param xI: Position of particle I (NumPy array)
    :param xJ: Position of particle J (NumPy array)
    :param h: Smoothing length
    :return: Kernel value W
    """
    dx = xI[0] - xJ[0]
    dy = xI[1] - xJ[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    q = r / h
    A = 40.0 / (h**2 * 7 * np.pi)

    if q > 1:
        return 0.0
    elif q > 0.5:
        return A * (2 * (1 - q)**3)
    else:
        return A * (6 * (q**3 - q**2) + 1)


@njit
def grad_m4_numba(xI, xJ, h):
    """
    Computes the gradient of the M4 kernel in 2D.
    
    The gradient is given by:
        ∇W(x, h) = (dW/dr) * ( (xI - xJ) / r ),
        
    where
        dW/dr = (1/h) * (dW/dq)  and q = r/h.
    
    For 0 <= q <= 0.5:
        dW/dq = A*(18*q^2 - 12*q),
    For 0.5 < q <= 1:
        dW/dq = -6*A*(1 - q)^2,
    and for q > 1, the gradient is zero.
    
    :param xI: Position of particle I (NumPy array)
    :param xJ: Position of particle J (NumPy array)
    :param h: Smoothing length
    :return: Gradient of the kernel as a NumPy array.
    """
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
        dW_dq = -6.0 * A * (1 - q)**2
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
