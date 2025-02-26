import numpy as np
from dfsph.grid import Grid
from dfsph.particles import Particles  # Now directly using SoA structure
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
        self.surface_tension_coeff = surface_tension_coeff  # New parameter

        self.mean_density = 0
        self.gamma_mass_solid = 1.4

        self.grid = Grid(grid_origin, grid_size, cell_size)
        self.gravity = np.array([0, -9.81], dtype=np.float64)

        # Initialize neighbors using the new grid structure
        self.find_neighbors()
        self.compute_density_and_alpha()
        self.update_mass_solid()

    def compute_density_and_alpha(self):
        """
        Compute density and alpha values for all particles using the Numba-optimized kernel functions.
        """
        densities, alphas = sphjit.compute_density_alpha_numba(
            self.particles.position, self.particles.mass,
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.rest_density)

        self.particles.density[:] = densities
        self.particles.alpha[:] = alphas
        self.mean_density = np.mean(densities)

    def compute_pressure(self, B=1000.0, gamma=7):
        """
        Compute pressure values using the Tait equation of state.
        """
        self.particles.pressure[:] = B * (
            (self.particles.density / self.rest_density)**gamma - 1)

    def update_mass_solid(self):
        """
        Adjust solid particle masses based on surrounding fluid particles.
        """
        new_masses = sphjit.update_mass_solid_numba(
            self.particles.position,
            (self.particles.types == 1).astype(np.int32),
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.rest_density,
            self.gamma_mass_solid, self.particles.mass.copy())

        self.particles.mass[self.particles.types == 1] = new_masses[
            self.particles.types == 1]

    def compute_viscosity_forces_updated(self):
        """
        Compute viscosity forces using an updated viscosity model.
        """
        vis_forces = sphjit.compute_viscosity_forces_updated_numba(
            self.particles.position,
            self.particles.velocity,
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

    def compute_pressure_forces_updated(self):
        """
        Compute pressure forces for all particles.
        """
        p_forces = sphjit.compute_pressure_forces_updated_numba(
            self.particles.position,
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.pressure,
            self.particles.mass, self.particles.neighbor_indices,
            self.particles.neighbor_starts, self.particles.neighbor_counts,
            self.h)

        self.particles.pressure_forces[:] = p_forces

    def compute_surface_tension_forces_updated(self):
        """
        Compute surface tension forces for fluid-fluid interactions.
        """
        surf_forces = sphjit.compute_surface_tension_forces_updated_numba(
            self.particles.position, self.particles.velocity,
            (self.particles.types == 1).astype(np.int32),
            self.particles.density, self.particles.mass,
            self.particles.neighbor_indices, self.particles.neighbor_starts,
            self.particles.neighbor_counts, self.h, self.surface_tension_coeff)

        self.particles.surface_tension_forces[:] = surf_forces

    def predict_intermediate_velocity(self):
        """
        Apply forces and update velocity predictions.
        """
        self.particles.external_forces[:] = self.particles.mass[:, np.
                                                                newaxis] * self.gravity

        # Compute total force and update velocity
        total_force = (self.particles.viscosity_forces +
                       self.particles.pressure_forces +
                       self.particles.surface_tension_forces +
                       self.particles.external_forces)
        acceleration = total_force / self.particles.mass[:, np.newaxis]
        self.particles.velocity[:] += acceleration * self.dt

    def integrate(self):
        """
        Integrate particle positions using velocity.
        """
        self.particles.position[:] += self.particles.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        """
        Enforce boundary conditions by damping velocity when particles reach the domain limits.
        """
        bottom, top = self.get_bottom_and_top()
        damping_factor = -collider_damping

        for i in range(self.num_particles):
            if self.particles.types[i] != 0:  # Skip non-fluid particles
                continue
            for d in range(2):
                if self.particles.position[i, d] < bottom[d]:
                    self.particles.position[i, d] = bottom[d]
                    self.particles.velocity[i, d] *= damping_factor
                elif self.particles.position[i, d] > top[d]:
                    self.particles.position[i, d] = top[d]
                    self.particles.velocity[i, d] *= damping_factor

    def get_bottom_and_top(self):
        """
        Get the bottom-left and top-right boundaries of the simulation.
        """
        left = self.grid.grid_origin[0]
        low = self.grid.grid_origin[1]
        right = left + self.grid.grid_size[0]
        high = low + self.grid.grid_size[1]
        return np.array([left, low],
                        dtype=np.float64), np.array([right, high],
                                                    dtype=np.float64)

    def find_neighbors(self):
        """
        Find neighbors for each particle using the grid system.
        Updates the neighbor data structure in Particles directly.
        """
        self.grid.find_neighbors(self.particles, self.h)

    def adapt_dt_for_cfl(self):
        """
        Adjust time step based on the CFL condition.
        """
        vmax = np.max(np.linalg.norm(self.particles.velocity, axis=1))
        if vmax < 1e-6:
            self.dt = 0.033
        else:
            self.dt = max(1e-4, min(0.3999 * self.h / vmax, 0.033))

    def reset_forces(self):
        """
        Reset all force accumulators to zero.
        """
        self.particles.viscosity_forces.fill(0)
        self.particles.external_forces.fill(0)
        self.particles.pressure_forces.fill(0)
        self.particles.surface_tension_forces.fill(0)

    def update(self):
        """
        Perform a full simulation step.
        """
        self.adapt_dt_for_cfl()
        self.reset_forces()
        self.find_neighbors()

        self.compute_density_and_alpha()
        self.update_mass_solid()
        self.compute_viscosity_forces_updated()
        self.compute_pressure()
        self.compute_pressure_forces_updated()
        self.compute_surface_tension_forces_updated()

        self.predict_intermediate_velocity()
        self.integrate()
        self.apply_boundary_penalty()
