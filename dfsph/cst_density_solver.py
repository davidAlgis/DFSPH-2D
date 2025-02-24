import numpy as np
from numba import njit, prange
from dfsph.kernels import cubic_kernel_numba, cubic_grad_kernel_numba


@njit(parallel=True)
def compute_density_error_numba(positions, velocities, masses, densities,
                                neighbor_indices, neighbor_starts,
                                neighbor_counts, h):
    n = positions.shape[0]
    density_errors = np.empty(n, dtype=np.float64)
    for i in prange(n):
        error = 0.0
        start = neighbor_starts[i]
        count = neighbor_counts[i]
        for k in range(count):
            j = neighbor_indices[start + k]
            # Compute difference in velocities
            dv0 = velocities[i, 0] - velocities[j, 0]
            dv1 = velocities[i, 1] - velocities[j, 1]
            # Compute kernel gradient (Numba-accelerated)
            grad_wij = cubic_grad_kernel_numba(positions[i], positions[j], h)
            error += masses[j] * (dv0 * grad_wij[0] + dv1 * grad_wij[1])
        density_errors[i] = error
    return density_errors


@njit(parallel=True)
def compute_velocity_correction_numba(positions, densities, masses, kappa,
                                      neighbor_indices, neighbor_starts,
                                      neighbor_counts, h):
    n = positions.shape[0]
    vel_corr = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        corr = np.zeros(2, dtype=np.float64)
        start = neighbor_starts[i]
        count = neighbor_counts[i]
        for k in range(count):
            j = neighbor_indices[start + k]
            denom = densities[i] + densities[j]
            if denom < 1e-8:
                continue
            grad_wij = cubic_grad_kernel_numba(positions[i], positions[j], h)
            corr[0] += masses[j] * (
                (kappa[i] + kappa[j]) / denom) * grad_wij[0]
            corr[1] += masses[j] * (
                (kappa[i] + kappa[j]) / denom) * grad_wij[1]
        vel_corr[i, :] = corr
    return vel_corr
