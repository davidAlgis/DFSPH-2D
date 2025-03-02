import numpy as np
from numba import njit, prange
from dfsph.kernels import grad_w


@njit(parallel=True)
def compute_intermediate_density_numba(density, position, velocity, mass,
                                       types, density_intermediate,
                                       neighbor_starts, neighbor_counts,
                                       neighbor_indices, dt, h, rest_density):
    n = density.shape[0]
    # density_intermediate = np.copy(density)
    for i in prange(n):
        if types[i] == 0:  # Only fluid particles
            sum_fluid = 0.0
            sum_solid = 0.0
            for idx in range(neighbor_starts[i],
                             neighbor_starts[i] + neighbor_counts[i]):
                j = neighbor_indices[idx]
                if i == j:
                    continue
                grad = grad_w(position[i], position[j], h)
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
        if types[i] == 0:
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


@njit(parallel=True)
def compute_density_derivative_numba(density, position, velocity, mass, types,
                                     density_derivative, neighbor_starts,
                                     neighbor_counts, neighbor_indices, dt, h,
                                     rest_density):
    n = density.shape[0]
    for i in prange(n):
        if types[i] != 0:
            continue
        # For fluid particles, compute derivative
        sum_deriv = 0.0
        for idx in range(neighbor_starts[i],
                         neighbor_starts[i] + neighbor_counts[i]):
            j = neighbor_indices[idx]
            gradwij = grad_w(position[i], position[j], h)
            # Compute relative velocity
            vij0 = velocity[i, 0] - velocity[j, 0]
            vij1 = velocity[i, 1] - velocity[j, 1]
            sum_deriv += mass[j] * (vij0 * gradwij[0] + vij1 * gradwij[1])

        density_derivative[i] = max(0, sum_deriv)
    return density_derivative


@njit(parallel=True)
def adapt_velocity_divergence_free_numba(position, velocity, density, alpha,
                                         mass, types, density_derivative,
                                         neighbor_starts, neighbor_counts,
                                         neighbor_indices, dt, h,
                                         rest_density):
    n = position.shape[0]
    # For each fluid particle, update its velocity to reduce divergence.
    for i in prange(n):
        if types[i] != 0:
            continue
        rho_i = density[i]
        # Compute kappa for divergence correction using the density derivative
        # For simplicity, we use alpha and the (precomputed) density derivative.
        # Here, we assume that compute_density_derivative_numba has been called
        # and the result stored externally, but for now, we compute on the fly.
        density_deriv = 0.0
        for idx in range(neighbor_starts[i],
                         neighbor_starts[i] + neighbor_counts[i]):
            j = neighbor_indices[idx]
            gradwij = grad_w(position[i], position[j], h)
            vij0 = velocity[i, 0] - velocity[j, 0]
            vij1 = velocity[i, 1] - velocity[j, 1]
            density_deriv += mass[i] * (vij0 * gradwij[0] + vij1 * gradwij[1])
        kappaDivergenceI = density_deriv * alpha[i] / dt
        kappaDivergenceIDivRhoI = kappaDivergenceI / rho_i
        vel_corr0 = 0.0
        vel_corr1 = 0.0
        for idx in range(neighbor_starts[i],
                         neighbor_starts[i] + neighbor_counts[i]):
            j = neighbor_indices[idx]
            gradwij = grad_w(position[i], position[j], h)
            if types[i] == 0:
                if types[j] == 0:
                    rho_j = density[j]
                    density_deriv_j = 0.0
                    for idx2 in range(neighbor_starts[j],
                                      neighbor_starts[j] + neighbor_counts[j]):
                        k = neighbor_indices[idx2]
                        if j == k:
                            continue
                        grad2 = grad_w(position[j], position[k], h)
                        diff0 = velocity[j, 0] - velocity[k, 0]
                        diff1 = velocity[j, 1] - velocity[k, 1]
                        density_deriv_j += mass[j] * (diff0 * grad2[0] +
                                                      diff1 * grad2[1])
                    kappaDivergenceJ = density_deriv_j * alpha[j] / dt
                    kappaDivergenceJDivRhoJ = kappaDivergenceJ / rho_j
                    vel_corr0 += mass[i] * (
                        kappaDivergenceIDivRhoI +
                        kappaDivergenceJDivRhoJ) * gradwij[0]
                    vel_corr1 += mass[i] * (
                        kappaDivergenceIDivRhoI +
                        kappaDivergenceJDivRhoJ) * gradwij[1]
                else:
                    vel_corr0 += (mass[j] /
                                  rho_i) * kappaDivergenceI * gradwij[0]
                    vel_corr1 += (mass[j] /
                                  rho_i) * kappaDivergenceI * gradwij[1]
        velocity[i, 0] -= dt * vel_corr0
        velocity[i, 1] -= dt * vel_corr1
