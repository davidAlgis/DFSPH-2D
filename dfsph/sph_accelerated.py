import numpy as np
from dfsph.kernels import cubic_kernel_numba, cubic_grad_kernel_numba
from numba import njit, prange


# Numba-accelerated functions
@njit(parallel=True)
def compute_density_alpha_numba(positions, masses, neighbor_indices,
                                neighbor_starts, neighbor_counts, h,
                                rest_density):
    n = positions.shape[0]
    densities = np.empty(n, dtype=np.float64)
    alphas = np.empty(n, dtype=np.float64)
    min_density = rest_density / 100.0

    for i in prange(n):
        density_fluid = 0.0
        sum_abs0 = 0.0
        sum_abs1 = 0.0
        abs_sum = 0.0

        start = neighbor_starts[i]
        count = neighbor_counts[i]

        for k in range(count):
            j = neighbor_indices[start + k]

            # Compute kernel and gradient using Numba-accelerated functions
            wij = cubic_kernel_numba(positions[i], positions[j], h)
            grad_wij = cubic_grad_kernel_numba(positions[i], positions[j], h)

            density_fluid += masses[j] * wij

            # Sum gradient terms for alpha computation
            sum_abs0 += masses[j] * grad_wij[0]
            sum_abs1 += masses[j] * grad_wij[1]
            abs_sum += (masses[j]**2) * (grad_wij[0]**2 + grad_wij[1]**2)

        densities[i] = max(density_fluid, min_density)
        norm = sum_abs0**2 + sum_abs1**2
        alphas[i] = density_fluid / (norm + abs_sum + 1e-5)

    return densities, alphas
