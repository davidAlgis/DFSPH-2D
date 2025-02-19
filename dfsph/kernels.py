import numpy as np

# Constants
CUBIC_SPLINE_FACTOR = 8.0 / np.pi
CUBIC_SPLINE_GRADIENT_FACTOR = 48.0 / np.pi


def kernel_cubic_spline_2d(xI, xJ, h):
    """
    2D Cubic Spline Kernel for SPH method. This kernel is based on the implementation 
    for a 0-1 interval cubic spline kernel in 2D.
    
    :param xI: Position of particle I (numpy array)
    :param xJ: Position of particle J (numpy array)
    :param h: The smoothing length (kernel radius)
    :return: The cubic spline kernel value between particles I and J
    """
    r = xI - xJ  # Distance vector between particles
    r_length = np.linalg.norm(r)  # Distance between particles
    one_div_h = 1.0 / h  # Inverse of smoothing length
    q = r_length * one_div_h  # Distance normalized by the smoothing length

    if q >= 2.0:
        return 0.0  # Kernel is zero if q >= 2.0

    # Cubic spline kernel function
    if q < 1.0:
        q_squared = q * q
        return CUBIC_SPLINE_FACTOR * (1 - 1.5 * q_squared +
                                      0.75 * q_squared * q) * one_div_h**2
    else:
        q_diff = 2 - q
        return CUBIC_SPLINE_FACTOR * q_diff**3 * one_div_h**2


def grad_kernel_cubic_spline_2d(xI, xJ, h):
    """
    2D Gradient of the Cubic Spline Kernel for SPH method. This function computes the gradient of 
    the cubic spline kernel for 2D, which is used in the SPH pressure forces.
    
    :param xI: Position of particle I (numpy array)
    :param xJ: Position of particle J (numpy array)
    :param h: The smoothing length (kernel radius)
    :return: The gradient of the cubic spline kernel (numpy array)
    """
    r = xI - xJ  # Distance vector between particles
    r_length = np.linalg.norm(r)  # Distance between particles
    if r_length < 1e-5:
        return np.zeros(2)  # Avoid division by zero, return zero vector

    one_div_h = 1.0 / h  # Inverse of smoothing length
    q = r_length * one_div_h  # Distance normalized by the smoothing length

    if q >= 2.0:
        return np.zeros(2)  # Gradient is zero if q >= 2.0

    one_div_h_pow_4 = one_div_h**4  # (1/h)^4
    if q < 1.0:
        q_squared = q * q
        result = 3 * q - 2
        grad = result * q * one_div_h**2 * (r / r_length)
    else:
        q_diff = 2 - q
        grad = -q_diff**2 * one_div_h**2 * (r / r_length)

    return grad * CUBIC_SPLINE_GRADIENT_FACTOR * one_div_h_pow_4


# Test the kernels with dummy data (remove or modify for actual use)
if __name__ == '__main__':
    # Example positions for particles (xI, xJ)
    xI = np.array([1.0, 2.0])
    xJ = np.array([2.0, 3.0])
    h = 1.0  # Smoothing length

    # Compute kernel and gradient
    kernel_value = kernel_cubic_spline_2d(xI, xJ, h)
    gradient_value = grad_kernel_cubic_spline_2d(xI, xJ, h)

    # Print results
    print("Kernel value:", kernel_value)
    print("Gradient value:", gradient_value)
