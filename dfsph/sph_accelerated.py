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


@njit(parallel=True)
def update_mass_solid_numba(positions, is_solid, neighbor_indices,
                            neighbor_starts, neighbor_counts, h, rest_density,
                            gamma_mass_solid, masses_out):
    """
    For each solid particle, sum w(...) with neighboring solids.
    Then set mass = rest_density * gamma_mass_solid / sum_w if sum_w>1e-8.
    """
    n = positions.shape[0]
    for i in prange(n):
        if is_solid[i] == 1:
            start = neighbor_starts[i]
            count = neighbor_counts[i]
            sum_w_val = 0.0
            for k in range(count):
                j = neighbor_indices[start + k]
                if is_solid[j] == 1:  # neighbor also solid
                    sum_w_val += w(positions[i], positions[j], h)
            if sum_w_val > 1e-8:
                masses_out[i] = rest_density * gamma_mass_solid / sum_w_val
    return masses_out


@njit(parallel=True)
def compute_density_alpha_numba(positions, masses, neighbor_indices,
                                neighbor_starts, neighbor_counts, h,
                                rest_density):
    """
    Compute density & alpha for each particle using w(...) & grad_w(...).
    """
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
            wij = w(positions[i], positions[j], h)
            grad_wij = grad_w(positions[i], positions[j], h)
            density_fluid += masses[j] * wij
            sum_abs0 += masses[j] * grad_wij[0]
            sum_abs1 += masses[j] * grad_wij[1]
            abs_sum += (masses[j]**2) * (grad_wij[0]**2 + grad_wij[1]**2)

        density_fluid = max(density_fluid, min_density)
        densities[i] = density_fluid
        norm = sum_abs0**2 + sum_abs1**2
        alphas[i] = density_fluid / (norm + abs_sum + 1e-5)

    return densities, alphas


@njit(parallel=True)
def compute_pressure_forces_updated_numba(positions, is_solid, densities,
                                          pressures, masses, neighbor_indices,
                                          neighbor_starts, neighbor_counts, h):
    """
    Compute the pressure force for each fluid particle.
    If neighbor is fluid:
        F += -mass_i^2*pressure_j/(rho_i*rho_j)*grad_wij
    If neighbor is solid:
        F += -mass_i*mass_j*(pressure_i/(rho_i^2))*grad_wij
    """
    n = positions.shape[0]
    pressure_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        if is_solid[i] == 1:  # skip solids
            continue

        rho_i = densities[i]
        pi = pressures[i]
        mass_i = masses[i]
        rho_i_sq = rho_i * rho_i

        start = neighbor_starts[i]
        count = neighbor_counts[i]
        accum = np.zeros(2, dtype=np.float64)

        for k in range(count):
            j = neighbor_indices[start + k]
            grad_ij = grad_w(positions[i], positions[j], h)

            if is_solid[j] == 0:
                # neighbor fluid
                accum -= (mass_i**2 * pressures[j]) / (rho_i *
                                                       densities[j]) * grad_ij
            else:
                # neighbor solid
                accum -= (mass_i * masses[j] * (pi / rho_i_sq)) * grad_ij

        pressure_forces[i] = accum

    return pressure_forces


@njit(parallel=True)
def compute_viscosity_forces_updated_numba(positions, velocities, is_solid,
                                           densities, masses, neighbor_indices,
                                           neighbor_starts, neighbor_counts, h,
                                           water_viscosity,
                                           viscosity_coefficient_solid):
    """
    If neighbor j is fluid:
        denom = (||r_ij||^2 + 1e-5)*density_j
        force += water_viscosity*masses[j]*(v_dot_r/denom)*gradW_ij
    If neighbor j is solid:
        denom = (||r_ij||^2 + 1e-5)*density_i
        force -= viscosity_coefficient_solid*masses[j]*(v_dot_r/denom)*gradW_ij

    Multiply final by factor=8*(mass_i/density_i).
    Returns an (n,2) array of viscosity forces.
    """
    n = positions.shape[0]
    viscosity_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        if is_solid[i] == 1:
            continue

        accum = np.zeros(2, dtype=np.float64)
        rho_i = densities[i]
        m_i = masses[i]
        vel_i = velocities[i]

        start = neighbor_starts[i]
        count = neighbor_counts[i]

        for k in range(count):
            j = neighbor_indices[start + k]

            r_ij = positions[i] - positions[j]
            r2 = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + 1e-5
            v_ij = vel_i - velocities[j]
            v_dot_r = v_ij[0] * r_ij[0] + v_ij[1] * r_ij[1]

            grad_ij = grad_w(positions[i], positions[j], h)

            if is_solid[j] == 0:
                denom = r2 * densities[j] + 1e-5
                accum += water_viscosity * masses[j] * (v_dot_r /
                                                        denom) * grad_ij
            else:
                denom = r2 * rho_i + 1e-5
                accum -= (viscosity_coefficient_solid * masses[j] *
                          (v_dot_r / denom) * grad_ij)

        factor = 8.0 * (m_i / (rho_i + 1e-5))
        accum *= factor
        viscosity_forces[i] = accum

    return viscosity_forces


@njit(parallel=True)
def compute_surface_tension_forces_updated_numba(
        positions, velocities, is_solid, densities, masses, neighbor_indices,
        neighbor_starts, neighbor_counts, h, surface_tension_coeff):
    """
    Simple model:
      F_surf = -sigma * sum_j( (mass_j / rho_j) * gradW_ij )
    for fluid-fluid interactions only.

    Return (n,2) array with surface tension for each fluid particle.
    """
    n = positions.shape[0]
    surf_forces = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        if is_solid[i] == 1:
            continue
        accum = np.zeros(2, dtype=np.float64)
        start = neighbor_starts[i]
        count = neighbor_counts[i]
        for k in range(count):
            j = neighbor_indices[start + k]
            if is_solid[j] == 0:  # fluid neighbor
                grad_ij = grad_w(positions[i], positions[j], h)
                accum -= surface_tension_coeff * (
                    masses[j] / (densities[j] + 1e-5)) * grad_ij

        surf_forces[i] = accum
    return surf_forces
