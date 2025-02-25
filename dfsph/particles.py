import numpy as np


class Particles:

    def __init__(self, num_particles):
        self.num_particles = num_particles

        # Particle properties stored as arrays
        # (x, y)
        self.position = np.zeros((num_particles, 2), dtype=np.float32)
        # (vx, vy)
        self.velocity = np.zeros((num_particles, 2), dtype=np.float32)

        self.viscosity_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.external_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.pressure_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.surface_tension_forces = np.zeros((num_particles, 2),
                                               dtype=np.float32)

        self.mass = np.ones(num_particles, dtype=np.float32)
        # Custom coefficient
        self.alpha = np.zeros(num_particles, dtype=np.float32)

        # List of lists for neighbors
        self.neighbors = [[] for _ in range(num_particles)]
        # Type ID for each particle
        self.types = np.zeros(num_particles, dtype=np.int32)

    def __repr__(self):
        return f"Particles(num={self.num_particles})"
