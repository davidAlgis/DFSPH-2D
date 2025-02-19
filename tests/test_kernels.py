import pytest
import numpy as np
from dfsph.kernels import kernel_cubic_spline, grad_kernel_cubic_spline

# We choose constants for the 2D cubic spline kernel as often found in literature.
# Here we use the form:
#   W(r, h) = (10 / (7π h^2)) * { 1 - 1.5*q^2 + 0.75*q^3    if 0<=q<1,
#                                  0.25*(2 - q)^3           if 1<=q<2,
#                                  0                        if q>=2 }
# where q = r/h.
#
# Adjust your kernel implementation accordingly if needed.


def test_kernel_normalization():
    """
    Verify that the kernel integrates approximately to 1 over its support.
    In 2D, we require:
      ∫_0^(2h) W(r, h) 2πr dr ≈ 1.
    We perform a discrete summation in Cartesian coordinates.
    """
    h = 1.0
    num_steps = 100
    x = np.linspace(-2 * h, 2 * h, num_steps)
    y = np.linspace(-2 * h, 2 * h, num_steps)
    dx = (4 * h) / (num_steps - 1)
    integral = 0.0
    for xi in x:
        for yi in y:
            r = np.sqrt(xi * xi + yi * yi)
            if r < 2 * h:
                # Evaluate the kernel with xI = [xi, yi] and xJ = [0,0]
                integral += kernel_cubic_spline(np.array([xi, yi]),
                                                np.array([0.0, 0.0]), h)
    integral *= dx * dx
    expected = 1.0
    # Allow a tolerance of 0.1 for numerical integration
    assert abs(integral - expected) < 0.1, \
        f"Normalization condition failed: Integral = {integral}, expected {expected}"


def test_kernel_positivity():
    """
    Verify that the kernel never returns a negative value.
    """
    h = 1.0
    for _ in range(100):
        xI = np.random.uniform(-2 * h, 2 * h, 2)
        xJ = np.random.uniform(-2 * h, 2 * h, 2)
        w = kernel_cubic_spline(xI, xJ, h)
        assert w >= 0, f"Kernel returned negative value: {w}"


def test_kernel_symmetry():
    """
    Verify that the kernel is symmetric:
      W(xI, xJ, h) == W(xJ, xI, h)
    """
    h = 1.0
    for _ in range(100):
        xI = np.random.uniform(-h, h, 2)
        xJ = np.random.uniform(-h, h, 2)
        w1 = kernel_cubic_spline(xI, xJ, h)
        w2 = kernel_cubic_spline(xJ, xI, h)
        assert abs(w1 - w2) < 1e-5, f"Kernel symmetry failed: {w1} != {w2}"


def test_kernel_compact_support():
    """
    Verify that the kernel is zero outside its support (r >= 2h).
    """
    h = 1.0
    # Choose points with distance >= 2h from the origin.
    xI = np.array([3.0 * h, 0.0])
    xJ = np.array([0.0, 0.0])
    w = kernel_cubic_spline(xI, xJ, h)
    assert abs(w) < 1e-5, f"Kernel not compact: W({xI}, {xJ}) = {w}"


def test_kernel_gradient_analytical_vs_numerical():
    """
    Compare the analytical gradient of the kernel with a finite-difference approximation.
    """
    h = 1.0
    epsilon = 1e-4
    for _ in range(50):
        xI = np.random.uniform(-h, h, 2)
        xJ = np.array([0.0, 0.0])
        grad_analytical = grad_kernel_cubic_spline(xI, xJ, h)

        grad_numerical = np.zeros(2)
        for i in range(2):
            xI_pos = xI.copy()
            xI_neg = xI.copy()
            xI_pos[i] += epsilon
            xI_neg[i] -= epsilon
            w_pos = kernel_cubic_spline(xI_pos, xJ, h)
            w_neg = kernel_cubic_spline(xI_neg, xJ, h)
            grad_numerical[i] = (w_pos - w_neg) / (2 * epsilon)

        diff = np.linalg.norm(grad_analytical - grad_numerical)
        assert diff < 1e-1, f"Kernel gradient mismatch: Analytical={grad_analytical}, Numerical={grad_numerical}, diff={diff}"


if __name__ == "__main__":
    pytest.main()
