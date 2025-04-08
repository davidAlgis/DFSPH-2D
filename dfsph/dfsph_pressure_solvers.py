from dfsph.kernels import grad_w
from numba import njit, prange


@njit(parallel=True)
def compute_intermediate_density_numba(
    density,
    position,
    velocity,
    mass,
    types,
    density_intermediate,
    neighbor_starts,
    neighbor_counts,
    neighbor_indices,
    dt,
    h,
    rest_density,
    box,
    box_not,
):
    n = density.shape[0]
    for i in prange(n):
        # Skip disabled particles
        if types[i] == -1:
            continue
        if not box.is_inside(
            position[i, 0], position[i, 1]
        ) or box_not.is_inside(position[i, 0], position[i, 1]):
            continue
        if types[i] != 0:
            continue  # Only fluid particles are processed
        sum_component = 0.0
        for idx in range(
            neighbor_starts[i], neighbor_starts[i] + neighbor_counts[i]
        ):
            j = neighbor_indices[idx]
            # Skip disabled neighbors
            if types[j] == -1:
                continue
            grad = grad_w(position[i], position[j], h)
            vij0 = velocity[i, 0] - velocity[j, 0]
            vij1 = velocity[i, 1] - velocity[j, 1]
            velocity_diff = vij0 * grad[0] + vij1 * grad[1]
            sum_component += mass[j] * velocity_diff
        density_intermediate_i = density[i] + dt * sum_component
        density_intermediate[i] = max(rest_density, density_intermediate_i)


@njit(parallel=True)
def adapt_velocity_density_numba(
    position,
    velocity,
    density,
    alpha,
    density_intermediate,
    pressure_forces,
    mass,
    types,
    neighbor_starts,
    neighbor_counts,
    neighbor_indices,
    dt,
    h,
    rest_density,
    box,
    box_not,
):
    n = velocity.shape[0]
    dt2 = dt * dt
    for i in prange(n):
        if types[i] != 0:
            continue
        if not box.is_inside(
            position[i, 0], position[i, 1]
        ) or box_not.is_inside(position[i, 0], position[i, 1]):
            continue
        if neighbor_counts[i] < 7:
            continue
        p_i = (density_intermediate[i] - rest_density) * alpha[i] / dt2
        p_div_rho_i = p_i / density[i]
        force_pressure_corr0 = 0.0
        force_pressure_corr1 = 0.0
        for idx in range(
            neighbor_starts[i], neighbor_starts[i] + neighbor_counts[i]
        ):
            j = neighbor_indices[idx]
            # Skip disabled neighbors
            if types[j] == -1:
                continue
            gradwij = grad_w(position[i], position[j], h)
            # fluid particles
            if types[j] == 0:
                p_j = (density_intermediate[j] - rest_density) * alpha[j] / dt2
                p_j = max(0, p_j)
                scalar = mass[j] * (p_div_rho_i + p_j / density[j])
                force_pressure_corr0 += scalar * gradwij[0]
                force_pressure_corr1 += scalar * gradwij[1]
            # solid particles
            else:
                scalar = 2.0 * mass[j] * p_div_rho_i
                force_pressure_corr0 += scalar * gradwij[0]
                force_pressure_corr1 += scalar * gradwij[1]

        pressure_forces[i, 0] -= mass[i] * force_pressure_corr0
        pressure_forces[i, 1] -= mass[i] * force_pressure_corr1
        velocity[i, 0] -= dt * force_pressure_corr0
        velocity[i, 1] -= dt * force_pressure_corr1


@njit(parallel=True)
def compute_density_derivative_numba(
    density,
    position,
    velocity,
    mass,
    types,
    density_derivative,
    neighbor_starts,
    neighbor_counts,
    neighbor_indices,
    dt,
    h,
    rest_density,
    box,
    box_not,
):
    n = density.shape[0]
    for i in prange(n):
        # Skip disabled particles
        if types[i] == -1:
            continue
        if not box.is_inside(
            position[i, 0], position[i, 1]
        ) or box_not.is_inside(position[i, 0], position[i, 1]):
            continue
        if types[i] != 0:
            continue
        sum_deriv = 0.0
        for idx in range(
            neighbor_starts[i], neighbor_starts[i] + neighbor_counts[i]
        ):
            j = neighbor_indices[idx]
            # Skip disabled neighbors
            if types[j] == -1:
                continue
            gradwij = grad_w(position[i], position[j], h)
            vij0 = velocity[i, 0] - velocity[j, 0]
            vij1 = velocity[i, 1] - velocity[j, 1]
            sum_deriv += mass[j] * (vij0 * gradwij[0] + vij1 * gradwij[1])
        density_derivative[i] = max(0, sum_deriv)
    return density_derivative


@njit(parallel=True)
def adapt_velocity_divergence_free_numba(
    position,
    velocity,
    density,
    alpha,
    mass,
    types,
    density_derivative,
    pressure_forces,
    neighbor_starts,
    neighbor_counts,
    neighbor_indices,
    dt,
    h,
    rest_density,
    box,
    box_not,
):
    n = position.shape[0]
    for i in prange(n):
        # Skip disabled particles
        if types[i] == -1:
            continue
        if not box.is_inside(
            position[i, 0], position[i, 1]
        ) or box_not.is_inside(position[i, 0], position[i, 1]):
            continue
        if types[i] != 0:
            continue
        if neighbor_counts[i] < 7:
            continue
        rho_i = density[i]
        kappaI = density_derivative[i] * alpha[i] / dt
        kappaIDivRhoI = kappaI / rho_i
        force_pressure_corr0 = 0.0
        force_pressure_corr1 = 0.0
        for idx in range(
            neighbor_starts[i], neighbor_starts[i] + neighbor_counts[i]
        ):
            j = neighbor_indices[idx]
            # Skip disabled neighbors
            if types[j] == -1:
                continue
            gradwij = grad_w(position[i], position[j], h)
            if types[j] == 0:
                kappaJ = density_derivative[j] * alpha[j] / dt
                kappaJDivRhoJ = kappaJ / density[j]
                scalar = mass[j] * (kappaIDivRhoI + kappaJDivRhoJ)
                force_pressure_corr0 += scalar * gradwij[0]
                force_pressure_corr1 += scalar * gradwij[1]
            else:
                scalar = (mass[j] / rho_i) * kappaI
                force_pressure_corr0 += scalar * gradwij[0]
                force_pressure_corr1 += scalar * gradwij[1]

        pressure_forces[i, 0] = mass[i] * force_pressure_corr0
        pressure_forces[i, 1] = mass[i] * force_pressure_corr1
        velocity[i, 0] -= dt * force_pressure_corr0
        velocity[i, 1] -= dt * force_pressure_corr1