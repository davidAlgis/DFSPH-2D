import numpy as np


class Particles:

    def __init__(self, num_particles: int = 0):
        """
        A class representing a collection of particles for a 2D Smoothed Particle Hydrodynamics (SPH) simulation.
        """
        self.num_particles = num_particles

        # Initialize empty arrays to allow dynamic insertion
        self.position = np.zeros((num_particles, 2), dtype=np.float32)
        self.velocity = np.zeros((num_particles, 2), dtype=np.float32)

        self.viscosity_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.external_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.pressure_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.surface_tension_forces = np.zeros((num_particles, 2),
                                               dtype=np.float32)

        self.mass = np.ones(num_particles, dtype=np.float32)
        self.alpha = np.zeros(num_particles, dtype=np.float32)

        self.neighbors = [[] for _ in range(num_particles)]
        self.types = np.zeros(num_particles, dtype=np.int32)

    def add_particle(self,
                     position,
                     velocity=(0, 0),
                     mass=1.0,
                     alpha=0.0,
                     particle_type=0):
        """
        Adds a new particle to the system.
        
        :param position: Tuple (x, y) of the particle position.
        :param velocity: Tuple (vx, vy) of the particle velocity.
        :param mass: Mass of the particle.
        :param alpha: Custom coefficient.
        :param particle_type: Type ID of the particle.
        """
        self.num_particles += 1

        # Append new particle properties
        self.position = np.vstack(
            (self.position, np.array(position, dtype=np.float32)))
        self.velocity = np.vstack(
            (self.velocity, np.array(velocity, dtype=np.float32)))

        self.viscosity_forces = np.vstack(
            (self.viscosity_forces, np.zeros(2, dtype=np.float32)))
        self.external_forces = np.vstack(
            (self.external_forces, np.zeros(2, dtype=np.float32)))
        self.pressure_forces = np.vstack(
            (self.pressure_forces, np.zeros(2, dtype=np.float32)))
        self.surface_tension_forces = np.vstack(
            (self.surface_tension_forces, np.zeros(2, dtype=np.float32)))

        self.mass = np.append(self.mass, mass)
        self.alpha = np.append(self.alpha, alpha)

        self.types = np.append(self.types, particle_type)
        self.neighbors.append(
            [])  # Initialize empty neighbor list for the new particle

    def __repr__(self):
        return f"Particles(num={self.num_particles})"
