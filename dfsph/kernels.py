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


def w(xI, xJ, h):
    kernel_cubic_spline(xI, xJ, h)


def grad_w(xI, xJ, h):
    grad_kernel_cubic_spline(xI, xJ, h)
