import numpy as np
from dfsph.grid import Grid
from dfsph.particles import Particles
from dfsph.kernels import w, grad_w
from dfsph.init_helper import DFSPHInitConfig
import dfsph.sph_accelerated as sphjit
import dfsph.dfsph_pressure_solvers as dfsph_pressure_solvers
import dfsph.particles_loader as exporter

# Constants for force types
PRESSURE = 0
VISCOSITY = 1
EXTERNAL = 2
SURFACE_TENSION = 3


class DFSPHSim:

    def __init__(self,
                 particles: Particles,
                 config: DFSPHInitConfig,
                 export_path=""):
        self.particles = particles
        self.num_particles = particles.num_particles
        self.h = config.h
        self.dt = config.dt
        self.sim_time = 0.0
        self.last_export_time = 0.0
        self.rest_density = config.rest_density
        self.water_viscosity = config.water_viscosity
        self.surface_tension_coeff = config.surface_tension_coeff
        self.export_path = export_path
        self.mean_density = 0
        self.gamma_mass_solid = 1.4
        if self.export_path:
            exporter.init_export(self.export_path)
            print(
                f"[Export] Data will be saved to '{self.export_path}' every 0.033 sec."
            )
        self.grid = Grid(config.grid_origin, config.grid_size,
                         config.cell_size)
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

    def compute_intermediate_density(self):
        """Wrapper that calls the Numba-parallelized intermediate density computation."""
        dfsph_pressure_solvers.compute_intermediate_density_numba(
            self.particles.density, self.particles.position,
            self.particles.velocity, self.particles.mass, self.particles.types,
            self.particles.density_intermediate,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.particles.neighbor_indices, self.dt, self.h,
            self.rest_density)

    def adapt_velocity_density(self):
        """Wrapper that calls the Numba-parallelized velocity adaptation."""
        dfsph_pressure_solvers.adapt_velocity_density_numba(
            self.particles.position, self.particles.velocity,
            self.particles.density, self.particles.alpha,
            self.particles.density_intermediate, self.particles.mass,
            self.particles.types, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.particles.neighbor_indices,
            self.dt, self.h, self.rest_density)

    def solve_constant_density(self):
        """Iteratively enforces incompressibility via density correction."""
        max_iter = 24
        for iteration in range(max_iter):
            self.compute_intermediate_density()

            # # Compute maximum density error for logging
            # density_errors = np.abs(self.particles.density_intermediate -
            #                         self.rest_density)
            fluid_mask = (self.particles.types == 0)
            density_errors = np.mean(
                np.abs(self.particles.density_intermediate[fluid_mask] -
                       self.rest_density))
            avg_error = np.mean(density_errors)

            if avg_error < 1e-3 * self.rest_density:
                break

            self.adapt_velocity_density()

        if iteration >= max_iter - 1:
            print(
                f"[Density Solver]: Max iteration reached ! avg error = {avg_error:.3f}"
            )

    def compute_density_derivative(self):
        """Calls the Numba function to compute density derivative for each particle."""
        dfsph_pressure_solvers.compute_density_derivative_numba(
            self.particles.density, self.particles.position,
            self.particles.velocity, self.particles.mass, self.particles.types,
            self.particles.density_derivative, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.particles.neighbor_indices,
            self.dt, self.h, self.rest_density)

    def adapt_velocity_divergence_free(self):
        """Calls the Numba function to adapt velocities for divergence-free constraint."""
        dfsph_pressure_solvers.adapt_velocity_divergence_free_numba(
            self.particles.position, self.particles.velocity,
            self.particles.density, self.particles.alpha, self.particles.mass,
            self.particles.types, self.particles.density_derivative,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.particles.neighbor_indices, self.dt, self.h,
            self.rest_density)

    def solve_divergence_free(self):
        """Iteratively enforces a divergence-free velocity field."""
        # Set divergence threshold and maximum iterations similar to C++ implementation
        threshold_divergence = 1e-3 * self.rest_density / self.dt
        max_iter = 24
        iter_count = 0
        density_derivative_avg = np.inf

        while (iter_count < max_iter) and (abs(density_derivative_avg)
                                           > threshold_divergence):
            # Update density derivative on each particle
            self.compute_density_derivative()

            # Adapt velocities to correct divergence
            self.adapt_velocity_divergence_free()

            # Compute average density derivative for fluid particles
            fluid_mask = (self.particles.types == 0)
            density_derivative_avg = np.mean(
                np.abs(self.particles.density_derivative[fluid_mask]))

            iter_count += 1
        if iter_count >= max_iter:
            print(
                f"[Divergence Solver]: Max iteration reached ! density derivative avg = {density_derivative_avg:.3f}"
            )

    def export_data(self):
        # **Handle exporting**
        if self.export_path and (self.sim_time -
                                 self.last_export_time) >= 0.033:
            exporter.export_snapshot(self.particles, self.export_path,
                                     self.sim_time)
            self.last_export_time = self.sim_time

    def update(self):
        self.adapt_dt_for_cfl()
        self.reset_forces()

        self.compute_viscosity_forces()
        # self.compute_surface_tension_forces()

        # Predict velocity based on external forces (fluid only)
        self.predict_intermediate_velocity()

        # Apply constant density solver
        self.solve_constant_density()

        # Integrate positions using the corrected velocities
        self.integrate()

        # Apply boundary conditions/penalties
        self.apply_boundary_penalty()

        # Update neighbor search and density/alpha after position update
        self.find_neighbors()
        self.compute_density_and_alpha()

        self.solve_divergence_free()
        self.sim_time += self.dt
        self.export_data()
