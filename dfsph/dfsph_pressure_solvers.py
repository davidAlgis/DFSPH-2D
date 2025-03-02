import numpy as np
from numba import njit, prange
from dfsph.kernels import grad_w  # Ensure grad_w is Numba‑compatible


@njit(parallel=True)
def compute_intermediate_density_numba(density, position, velocity, mass,
                                       types, neighbor_starts, neighbor_counts,
                                       neighbor_indices, dt, h, rest_density):
    n = density.shape[0]
    density_intermediate = np.copy(density)
    for i in prange(n):
        if types[i] == 0:  # Only fluid particles
            sum_fluid = 0.0
            sum_solid = 0.0
            for idx in range(neighbor_starts[i],
                             neighbor_starts[i] + neighbor_counts[i]):
                j = neighbor_indices[idx]
                if i == j:
                    continue
                # Compute kernel gradient between particle i and j.
                grad = grad_w(position[i], position[j], h)
                # Compute dot product between velocity difference and grad.
                diff0 = velocity[i, 0] - velocity[j, 0]
                diff1 = velocity[i, 1] - velocity[j, 1]
                velocity_diff = diff0 * grad[0] + diff1 * grad[1]
                if types[j] == 0:
                    sum_fluid += velocity_diff
                else:
                    sum_solid += mass[j] * velocity_diff
            temp = density[i] + dt * (mass[i] * sum_fluid + sum_solid)
            if temp < rest_density:
                density_intermediate[i] = rest_density
            else:
                density_intermediate[i] = temp
    return density_intermediate


@njit(parallel=True)
def adapt_velocity_density_numba(position, velocity, density, alpha,
                                 density_intermediate, mass, types,
                                 neighbor_starts, neighbor_counts,
                                 neighbor_indices, dt, h, rest_density):
    n = velocity.shape[0]
    for i in prange(n):
        if types[i] == 0:  # Only update fluid particles
            rho_i = density[i]
            kappa_i = (density_intermediate[i] -
                       rest_density) * alpha[i] / (dt * dt)
            kappa_div_rho_i = kappa_i / rho_i
            vel_corr0 = 0.0
            vel_corr1 = 0.0
            for idx in range(neighbor_starts[i],
                             neighbor_starts[i] + neighbor_counts[i]):
                j = neighbor_indices[idx]
                if i == j:
                    continue
                grad = grad_w(position[i], position[j], h)
                if types[j] == 0:
                    rho_j = density[j]
                    kappa_j = (density_intermediate[j] -
                               rest_density) * alpha[j] / (dt * dt)
                    vel_corr0 += mass[i] * (kappa_div_rho_i +
                                            kappa_j / rho_j) * grad[0]
                    vel_corr1 += mass[i] * (kappa_div_rho_i +
                                            kappa_j / rho_j) * grad[1]
                else:
                    vel_corr0 += 2.0 * mass[j] * kappa_div_rho_i * grad[0]
                    vel_corr1 += 2.0 * mass[j] * kappa_div_rho_i * grad[1]
            velocity[i, 0] -= dt * vel_corr0
            velocity[i, 1] -= dt * vel_corr1
