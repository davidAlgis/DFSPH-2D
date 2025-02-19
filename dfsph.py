import numpy as np
from grid import Grid
from particle import Particle, PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION


class DFSPHSim:

    def __init__(self, num_particles, h, mass, dt, grid_size, grid_position,
                 cell_size):
        """
        Initialize the DFSPH simulation with the given parameters.
        """
        self.num_particles = num_particles
        self.h = h
        self.mass = mass
        self.dt = dt

        # Initialize particles
        self.particles = self.initialize_particles()

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_size, grid_position, cell_size)

        # Physical parameters
        self.gravity = np.array([0, -9.81])  # Gravity force
        self.rest_density = 1000.0  # Example: rest density of the fluid
        self.viscosity = 0.1  # Example: viscosity coefficient

    def initialize_particles(self):
        """
        Initialize the particle data.
        Returns a list of Particle instances.
        """
        particles = []
        for _ in range(self.num_particles):
            position = np.random.rand(
                2) * 10  # Example: random position in a 10x10 domain
            velocity = np.zeros(2)
            particles.append(Particle(position, velocity, self.mass, self.h))
        return particles

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

    def update(self):
        """
        Perform a DFSPH update step, including:
        - Resetting forces
        - Applying external forces
        - Integrating velocities and positions
        """
        # Reset forces before computing them
        for particle in self.particles:
            particle.reset_forces()

        # Apply external forces (e.g., gravity)
        self.apply_external_forces()

        # Integrate positions and velocities
        self.integrate()

    def run(self, num_steps):
        """
        Run the simulation for a given number of time steps.

        :param num_steps: Number of simulation steps to run.
        """
        for step in range(num_steps):
            self.update()

            # Optional debug info
            if step % 100 == 0:
                print(f"Step {step}/{num_steps} complete.")
