import numpy as np
# I inspired myself from https://github.com/christophmschaefer/miluphcuda/blob/eab10e7eceb9cc6f833c37d491c9a8dd0a540424/kernel.cu
# Normalization factors for the cubic spline 0-1 kernel in 2D
CUBIC_SPLINE_01_FACTOR = 40.0 / (7 * np.pi)  # 2D normalization factor
CUBIC_SPLINE_01_GRADIENT_FACTOR = CUBIC_SPLINE_01_FACTOR * 6.0  # Factor for the gradient


def kernel_cubic_spline(xI, xJ, h):
    """
    Computes the 2D Cubic Spline Kernel for the 0-1 interval (used in SPH).

    Kernel formula:
        - If 0 <= q <= 0.5:  W = f * (6 * q^3 - 6 * q^2 + 1)
        - If 0.5 < q < 1:    W = 2 * f * (1 - q)^3
        - If q >= 1:         W = 0

    :param xI: Position of particle I (numpy array)
    :param xJ: Position of particle J (numpy array)
    :param h: The smoothing length (kernel radius)
    :return: Kernel value W
    """
    r = xI - xJ
    r_length = np.linalg.norm(r)  # Compute |r|
    q = r_length / h if r_length > 1e-5 else 0.0  # Normalize distance

    if q >= 1.0:
        return 0.0
    elif q <= 0.5:
        return CUBIC_SPLINE_01_FACTOR * (6 * q**3 - 6 * q**2 + 1)
    else:
        return 2 * CUBIC_SPLINE_01_FACTOR * (1 - q)**3


def grad_kernel_cubic_spline(xI, xJ, h):
    """
    Computes the gradient of the 2D Cubic Spline Kernel for the 0-1 interval.

    Gradient formula:
        - If 0 <= q <= 0.5:  dWdr = 6 * f / h * (3 * q^2 - 2 * q)
        - If 0.5 < q < 1:    dWdr = -6 * f / h * (1 - q)^2
        - If q >= 1:         dWdr = 0

    The gradient is computed as:
        ∇W = (dW/dr) * (r / |r|)

    :param xI: Position of particle I (numpy array)
    :param xJ: Position of particle J (numpy array)
    :param h: The smoothing length (kernel radius)
    :return: Gradient of the cubic spline kernel (numpy array)
    """
    r = xI - xJ
    r_length = np.linalg.norm(r)  # Compute |r|
    if r_length < 1e-5:
        return np.zeros(2)  # Avoid division by zero

    q = r_length / h

    dWdr = 0.0
    if q < 1.0:
        if q <= 0.5:
            dWdr = CUBIC_SPLINE_01_GRADIENT_FACTOR / h * (3 * q**2 - 2 * q)
        else:
            dWdr = -CUBIC_SPLINE_01_GRADIENT_FACTOR / h * (1 - q)**2

    # Compute gradient: grad(W) = (dW/dr) * (r / |r|)
    return (dWdr / r_length) * r


# another implementation of cubic spline kernel that comes from https://github.com/MmmmHeee/SPH-2D-Taichi/blob/master/src/smooth_kernel.py
__CUBIC_KERNEL_FACTOR = 10 / (7 * np.pi)


def cubic_kernel(xI, xJ, h):
    # Value of cubic spline smoothing kernel
    r = np.array(xI) - np.array(xJ)
    r_len = np.linalg.norm(r)

    half_h = h / 2
    k = __CUBIC_KERNEL_FACTOR / half_h**2
    q = r_len / half_h

    res = 0.0
    if q <= 1.0:
        q2 = q**2
        res = k * (1 - 1.5 * q2 + 0.75 * q * q2)
    elif q < 2.0:
        res = k * 0.25 * (2 - q)**3
    return res


def cubic_grad_kernel(xI, xJ, h):
    # Derivative of cubic spline smoothing kernel
    r = np.array(xI) - np.array(xJ)
    r_len = np.linalg.norm(r)
    r_dir = r / r_len if r_len != 0 else r

    half_h = h / 2
    k = __CUBIC_KERNEL_FACTOR / half_h**2
    q = r_len / half_h

    res = np.zeros_like(r, dtype=float)
    if q < 1.0:
        res = (k / half_h) * (-3. * q + 2.25 * q**2) * r_dir
    elif q < 2.0:
        res = -0.75 * (k / half_h) * (2. - q)**2 * r_dir
    return res


def w(xI, xJ, h):
    return cubic_kernel(xI, xJ, h)


def grad_w(xI, xJ, h):
    return cubic_grad_kernel(xI, xJ, h)
