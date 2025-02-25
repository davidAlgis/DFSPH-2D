import numpy as np
from dfsph.kernels import w, grad_w
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
            wij = w(positions[i], positions[j], h)
            density_fluid += masses[j] * wij
            
            # Sum gradient terms for alpha computation
            grad_wij = grad_w(positions[i], positions[j], h)
            sum_abs0 += masses[j] * grad_wij[0]
            sum_abs1 += masses[j] * grad_wij[1]
            abs_sum += (masses[j]**2) * (grad_wij[0]**2 + grad_wij[1]**2)

        density_fluid = max(density_fluid, min_density)
        densities[i] = density_fluid
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

            # Compute kernel gradient
            grad_wij = grad_w(positions[i], positions[j], h)
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

        viscosity_forces[i] = viscosity_factor * viscosity_force

    return viscosity_forces


# -----------------------------------------------------------------------------
# Compute pressure forces using Numba
@njit(parallel=True)
def compute_pressure_forces_wcsph(positions, pressures, densities, masses,
                                  neighbor_indices, neighbor_starts,
                                  neighbor_counts, h):
    n = positions.shape[0]
    pressure_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        force = np.zeros(2, dtype=np.float64)
        start = neighbor_starts[i]
        count = neighbor_counts[i]
        for k in range(count):
            j = neighbor_indices[start + k]
            grad = grad_w(positions[i], positions[j], h)
            # Sum the contribution from the neighbor using the SPH pressure force formulation.
            term = (pressures[i] / (densities[i] * densities[i]) +
                    pressures[j] / (densities[j] * densities[j]))
            force += masses[j] * term * grad
        # Pressure force is the negative gradient of pressure.
        pressure_forces[i, 0] = -force[0]
        pressure_forces[i, 1] = -force[1]

    return pressure_forces


@njit(parallel=True)
def update_mass_solid_numba(positions, is_solid, neighbor_indices,
                            neighbor_starts, neighbor_counts, h, rest_density,
                            gamma_mass_solid, masses_out):
    """
    For each solid particle, sum the w() contributions from neighbors that are also solid.
    Then set particle.mass = rest_density * gamma_mass_solid / sum_w
    If sum_w <= 1e-8, do nothing (mass remains the same).
    
    :param positions: (N,2) array of positions.
    :param is_solid: (N,) int array (1 if solid, 0 if fluid).
    :param neighbor_indices: 1D array of neighbor indices.
    :param neighbor_starts, neighbor_counts: For neighbor iteration.
    :param h: Smoothing length.
    :param rest_density: The base fluid rest density.
    :param gamma_mass_solid: Factor to update mass for solids.
    :param masses_out: (N,) array of masses to update in-place for solids.
    """
    n = positions.shape[0]

    for i in prange(n):
        if is_solid[i] == 1:  # Only update mass for solid
            start = neighbor_starts[i]
            count = neighbor_counts[i]
            sum_w_val = 0.0

            for k in range(count):
                j = neighbor_indices[start + k]
                if is_solid[j] == 1:  # neighbor also solid
                    sum_w_val += w(positions[i], positions[j], h)

            if sum_w_val > 1e-8:
                masses_out[i] = rest_density * gamma_mass_solid / sum_w_val
        # else fluid => do nothing
    return masses_out


@njit(parallel=True)
def compute_pressure_forces_updated_numba(positions, is_solid, densities,
                                          pressures, masses, neighbor_indices,
                                          neighbor_starts, neighbor_counts, h):
    """
    Compute the pressure force for each fluid particle.
    
    For each fluid particle i:
      - If neighbor j is fluid:
          F += -mass_i^2 * p_j / (rho_i*rho_j) * gradW_ij
      - If neighbor j is solid:
          F += -mass_i * mass_j * (p_i / rho_i^2) * gradW_ij
    
    Returns pressure_forces: (N, 2) array of the computed pressure force for each particle.
    """
    n = positions.shape[0]
    pressure_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        if is_solid[i] == 1:
            continue  # skip solids

        rho_i = densities[i]
        pi = pressures[i]
        mass_i = masses[i]
        rho_i_sq = rho_i * rho_i

        start = neighbor_starts[i]
        count = neighbor_counts[i]

        force_accum = np.zeros(2, dtype=np.float64)

        for k in range(count):
            j = neighbor_indices[start + k]
            grad_wij = grad_w(positions[i], positions[j], h)

            if is_solid[j] == 0:
                # neighbor is fluid
                force_accum -= (mass_i**2 * pressures[j] /
                                (rho_i * densities[j])) * grad_wij
            else:
                # neighbor is solid
                force_accum -= (mass_i * masses[j] *
                                (pi / rho_i_sq)) * grad_wij

        pressure_forces[i, 0] = force_accum[0]
        pressure_forces[i, 1] = force_accum[1]

    return pressure_forces


@njit(parallel=True)
def compute_viscosity_forces_updated_numba(positions, velocities, is_solid,
                                           densities, masses, neighbor_indices,
                                           neighbor_starts, neighbor_counts, h,
                                           water_viscosity,
                                           viscosity_coefficient_solid):
    """
    Compute viscosity force for each fluid particle i based on neighbors j.
    
    If neighbor j is fluid:
       denom = r^2*density_j + 1e-5
       force += water_viscosity * mass_j * (v_dot_r / denom)*gradW_ij
    If neighbor j is solid:
       denom = r^2*density_i + 1e-5
       force -= viscosity_coefficient_solid * mass_j*(v_dot_r/denom)*gradW_ij
    
    Multiply final force by factor = 8*(mass_i/density_i) and store in out array.
    """
    n = positions.shape[0]
    viscosity_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        if is_solid[i] == 1:
            continue  # skip solids

        force_accum = np.zeros(2, dtype=np.float64)
        mass_i = masses[i]
        rho_i = densities[i]
        vel_i = velocities[i]

        start = neighbor_starts[i]
        count = neighbor_counts[i]

        for k in range(count):
            j = neighbor_indices[start + k]
            r_ij = positions[i] - positions[j]
            r2 = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + 1e-5

            v_ij = vel_i - velocities[j]
            v_dot_r = v_ij[0] * r_ij[0] + v_ij[1] * r_ij[1]

            grad = grad_w(positions[i], positions[j], h)

            if is_solid[j] == 0:
                # neighbor fluid
                denom = r2 * densities[j] + 1e-5
                force_accum += water_viscosity * masses[j] * (v_dot_r /
                                                              denom) * grad
            else:
                # neighbor solid
                denom = r2 * rho_i + 1e-5
                force_accum -= viscosity_coefficient_solid * masses[j] * (
                    v_dot_r / denom) * grad

        factor = 8.0 * (mass_i / (rho_i + 1e-5))
        viscosity_forces[i] = factor * force_accum

    return viscosity_forces
