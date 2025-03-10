import numpy as np
from dfsph.grid import Grid
from dfsph.particles import Particles
from dfsph.kernels import w, grad_w
from dfsph.init_helper import DFSPHInitConfig
import dfsph.sph_accelerated as sphjit
import dfsph.dfsph_pressure_solvers as dfsph_pressure_solvers
import dfsph.particles_loader as exporter
from dfsph.box import Box

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

        # Initial neighbor search and density/alpha computation using the box.
        box = Box(self.grid.grid_origin[0], self.grid.grid_origin[1],
                  self.grid.grid_size[0], self.grid.grid_size[1])
        self.find_neighbors()
        self.compute_density_and_alpha(box)
        self.update_mass_solid(box)

    def compute_density_and_alpha(self, box: Box):
        densities, alphas = sphjit.compute_density_alpha_numba(
            self.particles.position.astype(np.float64), self.particles.mass,
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.rest_density, box)
        self.particles.density[:] = densities
        self.particles.alpha[:] = alphas
        self.mean_density = np.mean(densities)

    def compute_pressure_wcsph(self, B=1000.0, gamma=7, box: Box = None):
        if box is None:
            self.particles.pressure[:] = B * (
                (self.particles.density / self.rest_density)**gamma - 1)
        else:
            for i in range(self.num_particles):
                if box.is_inside(self.particles.position[i, 0],
                                 self.particles.position[i, 1]):
                    self.particles.pressure[i] = B * (
                        (self.particles.density[i] / self.rest_density)**gamma
                        - 1)

    def update_mass_solid(self, box: Box):
        new_masses = sphjit.update_mass_solid_numba(
            self.particles.position.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.rest_density,
            self.gamma_mass_solid, self.particles.mass.copy(), box)
        self.particles.mass[self.particles.types == 1] = new_masses[
            self.particles.types == 1]

    def compute_viscosity_forces(self, box: Box):
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
            0.01,  # Viscosity coefficient for solid
            box)
        self.particles.viscosity_forces[:] = vis_forces

    def compute_pressure_forces_wcsph(self, box: Box):
        p_forces = sphjit.compute_pressure_forces_updated_numba(
            self.particles.position.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.pressure,
            self.particles.mass, self.particles.neighbor_indices,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.h, box)
        self.particles.pressure_forces[:] = p_forces

    def compute_surface_tension_forces(self, box: Box):
        surf_forces = sphjit.compute_surface_tension_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.velocity.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.mass,
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.surface_tension_coeff,
            box)
        self.particles.surface_tension_forces[:] = surf_forces

    def predict_intermediate_velocity(self, box: Box):
        fluid_mask = (self.particles.types == 0)
        for i in range(self.num_particles):
            if not fluid_mask[i]:
                continue
            if not box.is_inside(self.particles.position[i, 0],
                                 self.particles.position[i, 1]):
                continue
            self.particles.external_forces[
                i] = self.particles.mass[i] * self.gravity
        total_force = (self.particles.viscosity_forces +
                       self.particles.pressure_forces +
                       self.particles.surface_tension_forces +
                       self.particles.external_forces)
        for i in range(self.num_particles):
            if not fluid_mask[i]:
                continue
            if not box.is_inside(self.particles.position[i, 0],
                                 self.particles.position[i, 1]):
                continue
            self.particles.velocity[i] += (total_force[i] /
                                           self.particles.mass[i]) * self.dt

    def integrate(self, box: Box):
        fluid_mask = (self.particles.types == 0)
        vel_max = np.zeros(2)
        vel_max[0] = 0
        vel_max[1] = -0.25 * self.grid.grid_size[1]
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            if not box.is_inside(self.particles.position[i, 0],
                                 self.particles.position[i, 1]):
                continue
            if self.particles.neighbor_counts[i] < 7:
                self.particles.velocity[i, 0] = vel_max[0]
                self.particles.velocity[i,
                                        1] = min(0, self.particles.velocity[i,
                                                                            1])
                vel_y_grav_only = self.particles.velocity[
                    i, 1] + self.dt * self.particles.external_forces[
                        i, 1] / self.particles.mass[i]
                self.particles.velocity[i, 1] = vel_y_grav_only
        self.particles.position[
            fluid_mask] += self.particles.velocity[fluid_mask] * self.dt

    def apply_boundary_penalty(self, box: Box, collider_damping=0.5):
        bottom, top = self.get_bottom_and_top()
        damping_factor = -collider_damping
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            if not box.is_inside(self.particles.position[i, 0],
                                 self.particles.position[i, 1]):
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

    def compute_intermediate_density(self, box: Box):
        dfsph_pressure_solvers.compute_intermediate_density_numba(
            self.particles.density, self.particles.position,
            self.particles.velocity, self.particles.mass, self.particles.types,
            self.particles.density_intermediate,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.particles.neighbor_indices, self.dt, self.h,
            self.rest_density, box)

    def adapt_velocity_density(self, box: Box):
        dfsph_pressure_solvers.adapt_velocity_density_numba(
            self.particles.position, self.particles.velocity,
            self.particles.density, self.particles.alpha,
            self.particles.density_intermediate, self.particles.mass,
            self.particles.types, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.particles.neighbor_indices,
            self.dt, self.h, self.rest_density, box)

    def solve_constant_density(self, box: Box):
        max_iter = 24
        for iteration in range(max_iter):
            self.compute_intermediate_density(box)
            fluid_mask = (self.particles.types == 0)
            density_errors = np.mean(
                np.abs(self.particles.density_intermediate[fluid_mask] -
                       self.rest_density))
            avg_error = np.mean(density_errors)
            if avg_error < 1e-3 * self.rest_density:
                break
            self.adapt_velocity_density(box)
        if iteration >= max_iter - 1:
            print(
                f"[Density Solver]: Max iteration reached! avg error = {avg_error:.3f}"
            )

    def compute_density_derivative(self, box: Box):
        dfsph_pressure_solvers.compute_density_derivative_numba(
            self.particles.density, self.particles.position,
            self.particles.velocity, self.particles.mass, self.particles.types,
            self.particles.density_derivative, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.particles.neighbor_indices,
            self.dt, self.h, self.rest_density, box)

    def adapt_velocity_divergence_free(self, box: Box):
        dfsph_pressure_solvers.adapt_velocity_divergence_free_numba(
            self.particles.position, self.particles.velocity,
            self.particles.density, self.particles.alpha, self.particles.mass,
            self.particles.types, self.particles.density_derivative,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.particles.neighbor_indices, self.dt, self.h,
            self.rest_density, box)

    def solve_divergence_free(self, box: Box):
        threshold_divergence = 1e-3 * self.rest_density / self.dt
        max_iter = 24
        iter_count = 0
        density_derivative_avg = np.inf
        while (iter_count < max_iter) and (abs(density_derivative_avg)
                                           > threshold_divergence):
            self.compute_density_derivative(box)
            self.adapt_velocity_divergence_free(box)
            fluid_mask = (self.particles.types == 0)
            density_derivative_avg = np.mean(
                np.abs(self.particles.density_derivative[fluid_mask]))
            iter_count += 1
        if iter_count >= max_iter:
            print(
                f"[Divergence Solver]: Max iteration reached! density derivative avg = {density_derivative_avg:.3f}"
            )

    def export_data(self):
        if self.export_path and (self.sim_time -
                                 self.last_export_time) >= 0.033:
            exporter.export_snapshot(self.particles, self.export_path,
                                     self.sim_time)
            self.last_export_time = self.sim_time

    def update(self):
        from dfsph.box import Box
        # Create a Box from the grid parameters.
        box = Box(self.grid.grid_origin[0], self.grid.grid_origin[1],
                  self.grid.grid_size[0], self.grid.grid_size[1])
        self.adapt_dt_for_cfl()
        self.reset_forces()

        self.compute_viscosity_forces(box)
        self.predict_intermediate_velocity(box)
        self.solve_constant_density(box)
        self.integrate(box)
        self.apply_boundary_penalty(box)
        self.find_neighbors()
        self.compute_density_and_alpha(box)
        self.solve_divergence_free(box)
        self.sim_time += self.dt
        self.export_data()
