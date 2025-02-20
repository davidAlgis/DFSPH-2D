import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
import dfsph.sph_accelerated


class DFSPHSim:

    def __init__(self,
                 particles,
                 h,
                 dt,
                 grid_size,
                 grid_position,
                 cell_size,
                 rest_density=1027,
                 water_viscosity=0.1):
        """
        Initialize the DFSPH simulation.
        """
        self.particles = particles
        self.num_particles = len(self.particles)
        self.h = h
        self.dt = dt
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.mean_density = 0

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_size, grid_position, cell_size)

        # Physical parameters
        self.gravity = np.array([0, -9.81])  # Gravity force

    def _pack_particle_data(self):
        """
        Pack particle data (positions, masses, velocities) into arrays.
        """
        N = self.num_particles
        positions = np.empty((N, 2), dtype=np.float64)
        velocities = np.empty((N, 2), dtype=np.float64)
        masses = np.empty(N, dtype=np.float64)
        neighbor_counts = np.empty(N, dtype=np.int32)
        neighbor_starts = np.empty(N, dtype=np.int32)

        # Collect neighbor indices
        neighbor_list = []
        for i, particle in enumerate(self.particles):
            positions[i, :] = particle.position
            velocities[i, :] = particle.velocity
            masses[i] = particle.mass
            n_neighbors = len(particle.neighbors)
            neighbor_counts[i] = n_neighbors
            neighbor_starts[i] = len(neighbor_list)
            for neighbor in particle.neighbors:
                neighbor_list.append(neighbor.index)

        neighbor_indices = np.array(neighbor_list, dtype=np.int32)
        return positions, velocities, masses, neighbor_indices, neighbor_starts, neighbor_counts

    def compute_density_and_alpha(self):
        """
        Compute the density and alpha coefficient for each particle using Numba.
        """
        positions, _, masses, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data(
        )

        densities, alphas = dfsph.sph_accelerated.compute_density_alpha_numba(
            positions, masses, neighbor_indices, neighbor_starts,
            neighbor_counts, self.h, self.rest_density)

        total_density = 0.0
        for i, particle in enumerate(self.particles):
            particle.density = densities[i]
            particle.alpha = alphas[i]
            total_density += densities[i]

        self.mean_density = total_density / self.num_particles

    def compute_viscosity_forces(self):
        """
        Compute the viscosity forces using Numba.
        """
        positions, velocities, masses, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data(
        )

        viscosity_forces = dfsph.sph_accelerated.compute_viscosity_forces(
            positions, velocities, self.get_particle_densities(), masses,
            neighbor_indices, neighbor_starts, neighbor_counts, self.h,
            self.water_viscosity)

        for i, particle in enumerate(self.particles):
            particle.add_force(VISCOSITY, viscosity_forces[i])

    def get_particle_densities(self):
        """
        Return an array of densities for the particles.
        """
        return np.array([p.density for p in self.particles], dtype=np.float64)

    def predict_intermediate_velocity(self):
        """
        Apply external forces (like gravity) to all particles.
        """
        for particle in self.particles:
            particle.add_force(EXTERNAL, particle.mass * self.gravity)
            total_force = particle.total_force()
            acceleration = total_force / particle.mass
            particle.velocity += acceleration * self.dt

    def integrate(self):
        """
        Integrate particle velocities and positions using Euler integration.
        """
        for particle in self.particles:
            particle.position += particle.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        """
        Apply penalty to particles when they hit the boundary of the grid.
        
        :param collider_damping: Damping factor for boundary interaction.
        """
        for particle in self.particles:
            for i in range(2):  # For 2D, apply penalty to x (0) and y (1)
                if particle.position[i] < self.grid.grid_position[i]:
                    particle.position[
                        i] = self.grid.grid_position[i] + self.h * (
                            self.grid.grid_position[i] - particle.position[i])
                    particle.velocity[i] *= -collider_damping
                elif particle.position[i] > self.grid.grid_position[
                        i] + self.grid.grid_size[i]:
                    particle.position[i] = (
                        self.grid.grid_position[i] + self.grid.grid_size[i]
                    ) - self.h * (
                        particle.position[i] -
                        (self.grid.grid_position[i] + self.grid.grid_size[i]))
                    particle.velocity[i] *= -collider_damping

    def find_neighbors(self):
        """
        Update each particle's neighbor list using the grid.
        """
        self.grid.update_grid(self.particles)
        for particle in self.particles:
            particle.neighbors = self.grid.find_neighbors(particle)

    def update(self):
        """
        Perform a DFSPH update step, including:
        - Resetting forces
        - Finding neighbors
        - Computing density and alpha using Numba
        - Applying external forces
        - Integrating velocities and positions
        - Applying boundary penalties
        Perform a DFSPH update step.
        """
        for particle in self.particles:
            particle.reset_forces()

        self.find_neighbors()
        self.compute_density_and_alpha()
        self.compute_viscosity_forces()
        self.predict_intermediate_velocity()
        self.integrate()
        self.apply_boundary_penalty()
