import numpy as np
from dfsph.kernels import cubic_kernel_numba, cubic_grad_kernel_numba
from numba import njit, prange


# -----------------------------------------------------------------------------
# Compute density and alpha using Numba
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


# -----------------------------------------------------------------------------
# Compute viscosity forces using Numba
@njit(parallel=True)
def compute_viscosity_forces(positions, velocities, densities, masses,
                             neighbor_indices, neighbor_starts,
                             neighbor_counts, h, water_viscosity):
    n = positions.shape[0]
    viscosity_forces = np.zeros((n, 2), dtype=np.float64)

    viscosity_factor = 2.0 * (2 + 2)  # 2*(d+2) for 2D case

    for i in prange(n):
        density_i = densities[i]
        viscosity_force = np.zeros(2, dtype=np.float64)

        start = neighbor_starts[i]
        count = neighbor_counts[i]

        for k in range(count):
            j = neighbor_indices[start + k]

            # Compute distance and kernel gradient
            grad_wij = cubic_grad_kernel_numba(positions[i], positions[j], h)
            r_ij = positions[i] - positions[j]
            r_len_sq = r_ij[0]**2 + r_ij[1]**2
            if r_len_sq < 1e-8:
                continue  # Avoid singularities

            # Compute velocity difference
            v_ij = velocities[i] - velocities[j]
            v_dot_x = v_ij[0] * r_ij[0] + v_ij[1] * r_ij[1]

            # Compute denominator with a small regularization term
            density_j = densities[j]
            denom = r_len_sq + 0.01 * h * h * density_j

            # Compute viscosity force
            viscosity_force += (masses[j] /
                                density_i) * water_viscosity * masses[j] * (
                                    v_dot_x / denom) * grad_wij

        # Store computed force
        viscosity_forces[i] = viscosity_factor * viscosity_force

    return viscosity_forces
