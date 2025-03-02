import numpy as np
from dfsph.grid import Grid
from dfsph.particles import Particles  # SoA structure
from dfsph.kernels import w, grad_w
import dfsph.sph_accelerated as sphjit

# Constants for force types
PRESSURE = 0
VISCOSITY = 1
EXTERNAL = 2
SURFACE_TENSION = 3


class DFSPHSim:

    def __init__(self,
                 particles: Particles,
                 h,
                 dt,
                 grid_origin,
                 grid_size,
                 cell_size,
                 rest_density=1027,
                 water_viscosity=10,
                 surface_tension_coeff=2.0):
        self.particles = particles
        self.num_particles = particles.num_particles
        self.h = h
        self.dt = dt
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.surface_tension_coeff = surface_tension_coeff

        self.mean_density = 0
        self.gamma_mass_solid = 1.4

        self.grid = Grid(grid_origin, grid_size, cell_size)
        self.gravity = np.array([0, -9.81], dtype=np.float64)

        # First update neighbors (which now updates all neighbor arrays)
        self.find_neighbors()
        self.compute_density_and_alpha()
        self.update_mass_solid()

    def compute_density_and_alpha(self):
        densities, alphas = sphjit.compute_density_alpha_numba(
            self.particles.position.astype(
                np.float64),  # ensure float64 for accuracy
            self.particles.mass,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.rest_density)
        self.particles.density[:] = densities
        self.particles.alpha[:] = alphas
        self.mean_density = np.mean(densities)

    def compute_pressure_wcsph(self, B=1000.0, gamma=7):
        self.particles.pressure[:] = B * (
            (self.particles.density / self.rest_density)**gamma - 1)

    def update_mass_solid(self):
        new_masses = sphjit.update_mass_solid_numba(
            self.particles.position.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.rest_density,
            self.gamma_mass_solid, self.particles.mass.copy())
        self.particles.mass[self.particles.types == 1] = new_masses[
            self.particles.types == 1]

    def compute_viscosity_forces(self):
        vis_forces = sphjit.compute_viscosity_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.velocity.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density,
            self.particles.mass,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.water_viscosity,
            0.01  # Viscosity coefficient for solid
        )
        self.particles.viscosity_forces[:] = vis_forces

    def compute_pressure_forces_wcsph(self):
        p_forces = sphjit.compute_pressure_forces_updated_numba(
            self.particles.position.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.pressure,
            self.particles.mass, self.particles.neighbor_indices,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.h)
        self.particles.pressure_forces[:] = p_forces

    def compute_surface_tension_forces(self):
        surf_forces = sphjit.compute_surface_tension_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.velocity.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.mass,
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.surface_tension_coeff)
        self.particles.surface_tension_forces[:] = surf_forces

    def predict_intermediate_velocity(self):
        """
        Update velocities only for fluid particles (type 0).
        Solid particles remain completely static.
        """
        # Create a boolean mask for fluid particles.
        fluid_mask = (self.particles.types == 0)
        # Set external forces only for fluid.
        self.particles.external_forces[fluid_mask] = (
            self.particles.mass[fluid_mask, np.newaxis] * self.gravity)
        # Compute total force (all forces are computed, but only update fluid).
        total_force = (self.particles.viscosity_forces +
                       self.particles.pressure_forces +
                       self.particles.surface_tension_forces +
                       self.particles.external_forces)
        # Update velocity only for fluid particles.
        self.particles.velocity[fluid_mask] += (
            total_force[fluid_mask] /
            self.particles.mass[fluid_mask, np.newaxis]) * self.dt

    def integrate(self):
        """
        Update positions only for fluid particles.
        Solid particles remain static.
        """
        fluid_mask = (self.particles.types == 0)
        self.particles.position[
            fluid_mask] += self.particles.velocity[fluid_mask] * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        bottom, top = self.get_bottom_and_top()
        damping_factor = -collider_damping
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            for d in range(2):
                if self.particles.position[i, d] < bottom[d]:
                    self.particles.position[i, d] = bottom[d]
                    self.particles.velocity[i, d] *= damping_factor
                elif self.particles.position[i, d] > top[d]:
                    self.particles.position[i, d] = top[d]
                    self.particles.velocity[i, d] *= damping_factor

    def get_bottom_and_top(self):
        left = self.grid.grid_origin[0]
        low = self.grid.grid_origin[1]
        right = left + self.grid.grid_size[0]
        high = low + self.grid.grid_size[1]
        return np.array([left, low],
                        dtype=np.float64), np.array([right, high],
                                                    dtype=np.float64)

    def find_neighbors(self):
        self.grid.find_neighbors(self.particles, self.h)

    def adapt_dt_for_cfl(self):
        vmax = np.max(np.linalg.norm(self.particles.velocity, axis=1))
        if vmax < 1e-6:
            self.dt = 0.033
        else:
            self.dt = max(1e-4, min(0.3999 * self.h / vmax, 0.033))

    def reset_forces(self):
        self.particles.viscosity_forces.fill(0)
        self.particles.external_forces.fill(0)
        self.particles.pressure_forces.fill(0)
        self.particles.surface_tension_forces.fill(0)

    # --- New Functions for Constant Density Solver ---

    def compute_intermediate_density(self):
        """Computes the predicted density before correction is applied."""
        density_intermediate = np.copy(self.particles.density)
        fluid_mask = (self.particles.types == 0)

        for i in np.where(fluid_mask)[0]:
            sum_fluid, sum_solid = 0.0, 0.0
            pos_i = self.particles.position[i]
            vel_i = self.particles.velocity[i]

            start = self.particles.neighbor_starts[i]
            count = self.particles.neighbor_counts[i]
            for idx in range(start, start + count):
                j = self.particles.neighbor_indices[idx]
                if i == j:
                    continue

                pos_j = self.particles.position[j]
                vel_j = self.particles.velocity[j]
                grad = grad_w(pos_i, pos_j, self.h)
                velocity_diff = np.dot(vel_i - vel_j, grad)

                if self.particles.types[j] == 0:  # Fluid neighbor
                    sum_fluid += velocity_diff
                else:  # Solid neighbor
                    sum_solid += self.particles.mass[j] * velocity_diff

            density_intermediate[i] = max(
                self.particles.density[i] + self.dt *
                (self.particles.mass[i] * sum_fluid + sum_solid),
                self.rest_density)
        return density_intermediate

    def adapt_velocity_density(self, density_intermediate):
        """Adjusts particle velocities to maintain constant density."""
        fluid_mask = (self.particles.types == 0)
        for i in np.where(fluid_mask)[0]:
            rho_i = self.particles.density[i]
            kappa_i = (density_intermediate[i] - self.rest_density
                       ) * self.particles.alpha[i] / (self.dt**2)
            kappa_div_rho_i = kappa_i / rho_i

            velocity_correction = np.zeros(2, dtype=np.float64)

            start = self.particles.neighbor_starts[i]
            count = self.particles.neighbor_counts[i]
            for idx in range(start, start + count):
                j = self.particles.neighbor_indices[idx]
                if i == j:
                    continue

                grad = grad_w(self.particles.position[i],
                              self.particles.position[j], self.h)

                if self.particles.types[j] == 0:  # Fluid neighbor
                    rho_j = self.particles.density[j]
                    kappa_j = (density_intermediate[j] - self.rest_density
                               ) * self.particles.alpha[j] / (self.dt**2)
                    velocity_correction += self.particles.mass[i] * (
                        kappa_div_rho_i + kappa_j / rho_j) * grad
                else:  # Solid neighbor
                    velocity_correction += 2.0 * self.particles.mass[
                        j] * kappa_div_rho_i * grad
            self.particles.velocity[i] -= self.dt * velocity_correction

    def solve_constant_density(self):
        """Iteratively enforces incompressibility via density correction."""
        for iteration in range(24):
            density_intermediate = self.compute_intermediate_density()

            # Compute maximum density error for logging
            density_errors = np.abs(density_intermediate - self.rest_density)
            max_error = np.max(density_errors)
            avg_error = np.mean(density_errors)

            print(
                f"[Density Solver] Iteration {iteration + 1}: max error = {max_error:.3f}, avg error = {avg_error:.3f}"
            )

            if max_error < 1e-3 * self.rest_density:
                break  # Converged

            self.adapt_velocity_density(density_intermediate)

    # --- Updated Update Method Based on C++ Loop ---

    def update(self):
        self.adapt_dt_for_cfl()
        self.reset_forces()
        # self.find_neighbors()
        self.compute_density_and_alpha()
        # self.update_mass_solid()
        self.compute_viscosity_forces()

        # Predict velocity based on external forces (fluid only)
        self.predict_intermediate_velocity()

        # Constant density solver replaces the traditional pressure force computation:
        self.solve_constant_density()

        # Integrate positions using the corrected velocities
        self.integrate()

        # Apply boundary conditions/penalties
        self.apply_boundary_penalty()

        # Update neighbor search and density/alpha after position update
        self.find_neighbors()
        self.compute_density_and_alpha()

        # (Optional) Placeholder for divergence correction step
        # self.correct_divergence_error()
