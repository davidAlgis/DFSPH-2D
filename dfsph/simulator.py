import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
from dfsph.kernels import w, grad_w


class DFSPHSim:

    def __init__(self,
                 particles,
                 h,
                 dt,
                 grid_size,
                 grid_position,
                 cell_size,
                 rest_density=1027):
        """
        Initialize the DFSPH simulation with the given parameters.
        
        :param particles: List of Particle instances.
        :param h: Support radius for SPH.
        :param dt: Time step.
        :param grid_size: Grid size (tuple).
        :param grid_position: Grid starting position (tuple).
        :param cell_size: Size of each grid cell.
        :param rest_density: Rest density of the fluid.
        """
        self.particles = particles
        self.num_particles = len(self.particles)
        self.h = h
        self.dt = dt
        self.rest_density = rest_density
        self.mean_density = 0

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_size, grid_position, cell_size)

        # Physical parameters
        self.gravity = np.array([0, -9.81])  # Gravity force
        self.viscosity = 0.1  # Example: viscosity coefficient

    def compute_density_and_alpha(self):
        """
        Compute the density and alpha coefficient for each particle.
        """
        min_density = self.rest_density / 100.0
        mean_density = 0
        for i, particle in enumerate(self.particles):

            density_fluid = 0.0
            sum_abs = np.zeros(2)
            abs_sum = 0.0
            for neighbor in particle.neighbors:

                wij = w(particle.position, neighbor.position, self.h)
                grad_wij = grad_w(particle.position, neighbor.position, self.h)

                # Fluid contribution
                density_fluid += neighbor.mass * wij
                sum_abs += neighbor.mass * grad_wij
                abs_sum += (neighbor.mass**2) * np.dot(grad_wij, grad_wij)

            # Store density (clamped to avoid division by zero)
            particle.density = max(min_density, density_fluid)
            mean_density += particle.density
            # Compute alpha (avoid division by zero)
            particle.alpha = density_fluid / max(
                1e-5,
                np.dot(sum_abs, sum_abs) + abs_sum)
        self.mean_density = mean_density / self.num_particles

    def apply_external_forces(self):
        """
        Apply external forces (like gravity) to all particles.
        """
        for particle in self.particles:
            particle.add_force(EXTERNAL, particle.mass * self.gravity)

    def integrate(self):
        """
        Integrate particle velocities and positions using Euler integration.
        """
        for particle in self.particles:
            total_force = particle.total_force()
            acceleration = total_force / particle.mass

            # Update velocity and position
            particle.velocity += acceleration * self.dt
            particle.position += particle.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        """
        Apply penalty to particles when they hit the boundary of the grid.
        
        :param collider_damping: Damping factor for boundary interaction.
        """
        for particle in self.particles:
            for i in range(2):  # For 2D, apply penalty to x (0) and y (1)
                if particle.position[i] < self.grid.grid_position[
                        i]:  # Lower bound
                    particle.position[
                        i] = self.grid.grid_position[i] + self.h * (
                            self.grid.grid_position[i] - particle.position[i])
                    particle.velocity[i] *= -collider_damping
                elif particle.position[i] > self.grid.grid_position[
                        i] + self.grid.grid_size[i]:  # Upper bound
                    particle.position[i] = (
                        self.grid.grid_position[i] + self.grid.grid_size[i]
                    ) - self.h * (
                        particle.position[i] -
                        (self.grid.grid_position[i] + self.grid.grid_size[i]))
                    particle.velocity[i] *= -collider_damping

    def find_neighbors(self):
        # Update the grid to store particles in their respective cells
        self.grid.update_grid(self.particles)
        for i, particle in enumerate(self.particles):
            neighbors = self.grid.find_neighbors(particle)
            particle.neighbors = neighbors

    def update(self):
        """
        Perform a DFSPH update step, including:
        - Resetting forces
        - Computing density and alpha
        - Applying external forces
        - Integrating velocities and positions
        - Applying boundary penalties
        """
        for particle in self.particles:
            particle.reset_forces()

        self.find_neighbors()

        # Compute density and alpha values for DFSPH
        self.compute_density_and_alpha()

        # Apply external forces (e.g., gravity)
        # self.apply_external_forces()

        # Integrate positions and velocities
        self.integrate()

        # Apply boundary penalties
        self.apply_boundary_penalty()
