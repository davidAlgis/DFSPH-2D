import dfsph.dfsph_pressure_solvers as dfsph_pressure_solvers
import dfsph.particles_loader as exporter
import dfsph.sph_accelerated as sphjit
import numpy as np
from dfsph.box import Box
from dfsph.grid import Grid
from dfsph.init_helper import DFSPHInitConfig
from dfsph.kernels import grad_w, w
from dfsph.particles import Particles

# Constants for force types
PRESSURE = 0
VISCOSITY = 1
EXTERNAL = 2
SURFACE_TENSION = 3


class DFSPHSim:
    def __init__(
        self, particles: Particles, config: DFSPHInitConfig, export_path=""
    ):
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
                f"[Export] Data will be saved to '{self.export_path}' every"
                f"0.033 sec."
            )
        self.grid = Grid(
            config.grid_origin, config.grid_size, config.cell_size
        )
        self.gravity = np.array([0, -9.81], dtype=np.float64)

        # Create a Box corresponding to the grid.
        box = Box(
            self.grid.grid_origin[0],
            self.grid.grid_origin[1],
            self.grid.grid_size[0],
            self.grid.grid_size[1],
        )
        # For now, we assume an empty "box_not" (a box that covers no area).
        box_not = Box(0.0, 0.0, 0.0, 0.0)
        self.find_neighbors()
        self.compute_density_and_alpha(box, box_not)
        self.update_mass_solid(box, box_not)

    def filter_disabled_neighbors(self):
        sphjit.filter_disabled_neighbors_numba(
            self.particles.types,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
        )

    def compute_density_and_alpha(self, box: Box, box_not: Box):
        densities, alphas = sphjit.compute_density_alpha_numba(
            self.particles.position.astype(np.float64),
            self.particles.mass,
            self.particles.types,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.rest_density,
            box,
            box_not,
        )
        active_mask = self.particles.types != -1
        self.particles.density[active_mask] = densities[active_mask]
        self.particles.alpha[active_mask] = alphas[active_mask]
        self.mean_density = np.mean(self.particles.density[active_mask])

    def compute_pressure_wcsph(
        self, B=1000.0, gamma=7, box: Box = None, box_not: Box = None
    ):
        if box is None or box_not is None:
            active_mask = self.particles.types != -1
            self.particles.pressure[active_mask] = B * (
                (self.particles.density[active_mask] / self.rest_density)
                ** gamma
                - 1
            )
        else:
            for i in range(self.num_particles):
                if self.particles.types[i] == -1:
                    continue
                if box.is_inside(
                    self.particles.position[i, 0],
                    self.particles.position[i, 1],
                ) and not box_not.is_inside(
                    self.particles.position[i, 0],
                    self.particles.position[i, 1],
                ):
                    self.particles.pressure[i] = B * (
                        (self.particles.density[i] / self.rest_density)
                        ** gamma
                        - 1
                    )

    def update_mass_solid(self, box: Box, box_not: Box):
        new_masses = sphjit.update_mass_solid_numba(
            self.particles.position.astype(np.float64),
            self.particles.types,
            (self.particles.types == 1).astype(np.int32),
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.rest_density,
            self.gamma_mass_solid,
            self.particles.mass.copy(),
            box,
            box_not,
        )
        solid_mask = self.particles.types == 1
        self.particles.mass[solid_mask] = new_masses[solid_mask]

    def compute_viscosity_forces(self, box: Box, box_not: Box):
        vis_forces = sphjit.compute_viscosity_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.velocity.astype(np.float64),
            self.particles.types,
            (self.particles.types == 1).astype(np.int32),
            self.particles.density,
            self.particles.mass,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.water_viscosity,
            0.01,  # Viscosity coefficient for solid
            box,
            box_not,
        )
        active_mask = self.particles.types != -1
        self.particles.viscosity_forces[active_mask] = vis_forces[active_mask]

    def compute_pressure_forces_wcsph(self, box: Box, box_not: Box):
        p_forces = sphjit.compute_pressure_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.types,
            (self.particles.types == 1).astype(np.int32),
            self.particles.density,
            self.particles.pressure,
            self.particles.mass,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            box,
            box_not,
        )
        active_mask = self.particles.types != -1
        self.particles.pressure_forces[active_mask] = p_forces[active_mask]

    def compute_surface_tension_forces(self, box: Box, box_not: Box):
        surf_forces = sphjit.compute_surface_tension_forces_updated_numba(
            self.particles.position.astype(np.float64),
            self.particles.types,
            self.particles.velocity.astype(np.float64),
            (self.particles.types == 1).astype(np.int32),
            self.particles.density,
            self.particles.mass,
            self.particles.neighbor_indices,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.h,
            self.surface_tension_coeff,
            box,
            box_not,
        )
        active_mask = self.particles.types != -1
        self.particles.surface_tension_forces[active_mask] = surf_forces[
            active_mask
        ]

    def predict_intermediate_velocity(self, box: Box, box_not: Box):
        fluid_mask = self.particles.types == 0
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            pos = self.particles.position[i]
            if not box.is_inside(pos[0], pos[1]) or box_not.is_inside(
                pos[0], pos[1]
            ):
                continue
            self.particles.external_forces[i] = (
                self.particles.mass[i] * self.gravity
            )

        total_force = (
            self.particles.viscosity_forces
            + self.particles.surface_tension_forces
            + self.particles.external_forces
        )
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            pos = self.particles.position[i]
            if not box.is_inside(pos[0], pos[1]) or box_not.is_inside(
                pos[0], pos[1]
            ):
                continue
            self.particles.velocity[i] += (
                total_force[i] / self.particles.mass[i]
            ) * self.dt

    def explosion_check(self, box: Box, box_not: Box):
        vel_max = np.array(
            [0, -0.25 * self.grid.grid_size[1]], dtype=np.float64
        )
        for i in range(self.num_particles):
            if self.particles.types[i] == -1:
                continue
            if self.particles.types[i] != 0:
                continue
            if not box.is_inside(
                self.particles.position[i, 0], self.particles.position[i, 1]
            ) or box_not.is_inside(
                self.particles.position[i, 0], self.particles.position[i, 1]
            ):
                continue
            if self.particles.neighbor_counts[i] < 7:
                self.particles.velocity[i, 0] = vel_max[0]
                self.particles.velocity[i, 1] = min(
                    0, self.particles.velocity[i, 1]
                )
                vel_y_grav_only = self.particles.velocity[i, 1] + self.dt * (
                    self.particles.external_forces[i, 1]
                    / self.particles.mass[i]
                )
                self.particles.velocity[i, 1] = vel_y_grav_only

    def integrate(self, box: Box, box_not: Box):
        self.explosion_check(box, box_not)

        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            self.particles.velocity[i] += (
                self.particles.pressure_forces[i] / self.particles.mass[i]
            ) * self.dt

            self.particles.position[i] += self.particles.velocity[i] * self.dt

    def apply_boundary_penalty(
        self, box: Box, box_not: Box, collider_damping=0.5
    ):
        bottom, top = self.get_bottom_and_top()
        damping_factor = -collider_damping
        for i in range(self.num_particles):
            if self.particles.types[i] == -1:
                continue
            if self.particles.types[i] != 0:
                continue
            if not box.is_inside(
                self.particles.position[i, 0], self.particles.position[i, 1]
            ) or box_not.is_inside(
                self.particles.position[i, 0], self.particles.position[i, 1]
            ):
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
        return np.array([left, low], dtype=np.float64), np.array(
            [right, high], dtype=np.float64
        )

    def find_neighbors(self):
        self.grid.find_neighbors(self.particles, self.h)
        self.filter_disabled_neighbors()

    def adapt_dt_for_cfl(self):
        active_velocities = self.particles.velocity[self.particles.types != -1]
        vmax = (
            np.max(np.linalg.norm(active_velocities, axis=1))
            if active_velocities.shape[0] > 0
            else 0
        )
        if vmax < 1e-6:
            self.dt = 0.033
        else:
            self.dt = max(1e-4, min(0.3999 * self.h / vmax, 0.033))

    def reset_forces(self):
        active_mask = self.particles.types != -1
        self.particles.viscosity_forces[active_mask] = 0
        self.particles.external_forces[active_mask] = 0
        self.particles.pressure_forces[active_mask] = 0
        self.particles.surface_tension_forces[active_mask] = 0

    def compute_intermediate_density(self, box: Box, box_not: Box):
        dfsph_pressure_solvers.compute_intermediate_density_numba(
            self.particles.density,
            self.particles.position,
            self.particles.velocity_intermediate,
            self.particles.mass,
            self.particles.types,
            self.particles.density_intermediate,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.particles.neighbor_indices,
            self.dt,
            self.h,
            self.rest_density,
            box,
            box_not,
        )

    def adapt_velocity_density(self, box: Box, box_not: Box):

        dfsph_pressure_solvers.adapt_velocity_density_numba(
            self.particles.position,
            self.particles.velocity_intermediate,
            self.particles.density,
            self.particles.alpha,
            self.particles.density_intermediate,
            self.particles.pressure_forces,
            self.particles.mass,
            self.particles.types,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.particles.neighbor_indices,
            self.dt,
            self.h,
            self.rest_density,
            box,
            box_not,
        )

    def compute_density_error(self, box: Box, box_not: Box):
        density_errors = 0
        nbr_particles_compute = 0
        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            pos = self.particles.position[i]
            if box.is_inside(pos[0], pos[1]) and not box_not.is_inside(
                pos[0], pos[1]
            ):
                density_errors += np.abs(
                    self.particles.density_intermediate[i] - self.rest_density
                )
                nbr_particles_compute += 1
        avg_error = density_errors / nbr_particles_compute
        return avg_error

    def solve_constant_density(self, box: Box, box_not: Box):
        max_iter = 24
        self.particles.velocity_intermediate = self.particles.velocity.copy()
        active_mask = self.particles.types == 0
        self.particles.pressure_forces[active_mask] = 0
        for iteration in range(max_iter):
            self.compute_intermediate_density(box, box_not)
            # compute density errors on particles inside box and not inside
            # box_not
            avg_error = self.compute_density_error(box, box_not)
            if avg_error < 1e-3 * self.rest_density:
                break
            self.adapt_velocity_density(box, box_not)
        if iteration >= max_iter - 1:
            print(
                f"[Density Solver]: Max iteration reached! avg error ="
                f"{avg_error:.3f}"
            )

    def compute_density_derivative(self, box: Box, box_not: Box):
        dfsph_pressure_solvers.compute_density_derivative_numba(
            self.particles.density,
            self.particles.position,
            self.particles.velocity_intermediate,
            self.particles.mass,
            self.particles.types,
            self.particles.density_derivative,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.particles.neighbor_indices,
            self.dt,
            self.h,
            self.rest_density,
            box,
            box_not,
        )

    def adapt_velocity_divergence_free(self, box: Box, box_not: Box):
        dfsph_pressure_solvers.adapt_velocity_divergence_free_numba(
            self.particles.position,
            self.particles.velocity_intermediate,
            self.particles.density,
            self.particles.alpha,
            self.particles.mass,
            self.particles.types,
            self.particles.density_derivative,
            self.particles.pressure_forces,
            self.particles.neighbor_starts,
            self.particles.neighbor_counts,
            self.particles.neighbor_indices,
            self.dt,
            self.h,
            self.rest_density,
            box,
            box_not,
        )

    def compute_divergence_error(self, box: Box, box_not: Box):
        density_derivative_errors = 0
        nbr_particles_compute = 0
        for i in range(self.num_particles):
            if self.particles.types[i] == -1:
                continue
            pos = self.particles.position[i]
            if box.is_inside(pos[0], pos[1]) and not box_not.is_inside(
                pos[0], pos[1]
            ):
                density_derivative_errors += np.abs(
                    self.particles.density_derivative[i]
                )
                nbr_particles_compute += 1
        density_derivative_avg = (
            density_derivative_errors / nbr_particles_compute
        )
        return density_derivative_avg

    def solve_divergence_free(self, box: Box, box_not: Box):
        threshold_divergence = 1e-3 * self.rest_density / self.dt
        max_iter = 24
        iter_count = 0
        density_derivative_avg = np.inf
        self.particles.velocity_intermediate = self.particles.velocity.copy()
        active_mask = self.particles.types == 0
        self.particles.pressure_forces[active_mask] = 0
        while (iter_count < max_iter) and (
            abs(density_derivative_avg) > threshold_divergence
        ):
            self.compute_density_derivative(box, box_not)
            self.adapt_velocity_divergence_free(box, box_not)
            # compute density errors on particles inside box and not inside
            # box_not
            density_derivative_avg = self.compute_divergence_error(
                box, box_not
            )
            iter_count += 1
        if iter_count >= max_iter:
            print(
                f"[Divergence Solver]: Max iteration reached! density"
                f"derivative avg = {density_derivative_avg:.3f}"
            )

    def export_data(self):
        if (
            self.export_path
            and (self.sim_time - self.last_export_time) >= 0.033
        ):
            exporter.export_snapshot(
                self.particles, self.export_path, self.sim_time
            )
            self.last_export_time = self.sim_time

    def integrate_velocity(self, box: Box, box_not: Box):
        self.explosion_check(box, box_not)

        for i in range(self.num_particles):
            if self.particles.types[i] != 0:
                continue
            self.particles.velocity[i] += (
                self.particles.pressure_forces[i] / self.particles.mass[i]
            ) * self.dt

    def update(self):
        # Create a Box from the grid parameters and an "empty" box_not.
        box = Box(
            self.grid.grid_origin[0],
            self.grid.grid_origin[1],
            self.grid.grid_size[0],
            self.grid.grid_size[1],
        )
        box_not = Box(0.0, 0.0, 0.0, 0.0)
        self.adapt_dt_for_cfl()
        self.reset_forces()

        self.compute_viscosity_forces(box, box_not)
        self.predict_intermediate_velocity(box, box_not)
        self.solve_constant_density(box, box_not)
        self.integrate(box, box_not)
        self.apply_boundary_penalty(box, box_not)
        self.find_neighbors()
        self.compute_density_and_alpha(box, box_not)
        self.solve_divergence_free(box, box_not)
        self.integrate_velocity(box, box_not)
        self.sim_time += self.dt
        self.export_data()
