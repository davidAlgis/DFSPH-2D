import numpy as np


class Particles:

    def __init__(self, num_particles: int = 0):
        """
        A class representing a collection of particles for a 2D Smoothed
        Particle Hydrodynamics (SPH) simulation. Uses a Structure of Arrays
        (SoA) layout for optimized computation.
        """
        self.num_particles = num_particles

        # Initialize particle properties
        self.position = np.zeros((num_particles, 2), dtype=np.float32)
        self.velocity = np.zeros((num_particles, 2), dtype=np.float32)

        self.viscosity_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.external_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.pressure_forces = np.zeros((num_particles, 2), dtype=np.float32)
        self.surface_tension_forces = np.zeros(
            (num_particles, 2), dtype=np.float32
        )

        self.mass = np.ones(num_particles, dtype=np.float32)
        self.density = np.ones(num_particles, dtype=np.float32)
        # for constant density solver
        self.density_intermediate = np.ones(num_particles, dtype=np.float32)
        # for divergence free solver
        self.density_derivative = np.ones(num_particles, dtype=np.float32)
        self.alpha = np.zeros(num_particles, dtype=np.float32)
        self.pressure = np.zeros(num_particles, dtype=np.float32)

        self.types = np.zeros(
            num_particles, dtype=np.int32
        )  # 0 = fluid, 1 = solid

        # Neighbor data for optimized access
        self.neighbor_indices = np.zeros(
            0, dtype=np.int32
        )  # Flat array of neighbor indices
        self.neighbor_counts = np.zeros(
            num_particles, dtype=np.int32
        )  # Number of neighbors per particle
        self.neighbor_starts = np.zeros(
            num_particles, dtype=np.int32
        )  # Start index per particle

    def add_particle(
        self, position, velocity=(0, 0), mass=1.0, alpha=0.0, particle_type=0
    ):
        """
        Adds a new particle to the system.

        :param position: Tuple (x, y) of the particle position.
        :param velocity: Tuple (vx, vy) of the particle velocity.
        :param mass: Mass of the particle.
        :param alpha: Custom coefficient.
        :param particle_type: Type ID of the particle (0 = fluid, 1 = solid).
        """
        i = self.num_particles
        self.num_particles += 1

        # Resize arrays to accommodate new particle
        self.position = np.resize(self.position, (self.num_particles, 2))
        self.velocity = np.resize(self.velocity, (self.num_particles, 2))

        self.viscosity_forces = np.resize(
            self.viscosity_forces, (self.num_particles, 2)
        )
        self.external_forces = np.resize(
            self.external_forces, (self.num_particles, 2)
        )
        self.pressure_forces = np.resize(
            self.pressure_forces, (self.num_particles, 2)
        )
        self.surface_tension_forces = np.resize(
            self.surface_tension_forces, (self.num_particles, 2)
        )

        self.mass = np.resize(self.mass, self.num_particles)
        self.density = np.resize(self.density, self.num_particles)
        self.alpha = np.resize(self.alpha, self.num_particles)
        self.pressure = np.resize(self.pressure, self.num_particles)
        # for constant density solver
        self.density_intermediate = np.resize(
            self.density_intermediate, self.num_particles
        )
        # for divergence free solver
        self.density_derivative = np.resize(
            self.density_derivative, self.num_particles
        )

        self.types = np.resize(self.types, self.num_particles)
        self.neighbor_counts = np.resize(
            self.neighbor_counts, self.num_particles
        )
        self.neighbor_starts = np.resize(
            self.neighbor_starts, self.num_particles
        )

        # Assign properties
        self.position[i] = position
        self.velocity[i] = velocity
        self.mass[i] = mass
        self.alpha[i] = alpha
        self.types[i] = particle_type

        # Initialize empty neighbor count
        self.neighbor_counts[i] = 0

    def update_neighbors(self, neighbors_list):
        """
        Converts a list-of-lists neighbor structure into an efficient SoA
format.
        :param neighbors_list: List of neighbor indices per particle (list of
        lists).
        """
        total_neighbors = sum(len(neigh) for neigh in neighbors_list)
        self.neighbor_indices = np.zeros(total_neighbors, dtype=np.int32)
        self.neighbor_counts = np.zeros(self.num_particles, dtype=np.int32)
        self.neighbor_starts = np.zeros(self.num_particles, dtype=np.int32)
        idx = 0
        for i, neigh in enumerate(neighbors_list):
            self.neighbor_starts[i] = idx
            self.neighbor_counts[i] = len(neigh)
            self.neighbor_indices[idx : idx + len(neigh)] = neigh
            idx += len(neigh)

    def remove_type(self, type_to_remove: int):
        """
        Removes all particles of the specified type from the system.

        This method filters out all particles whose `types` attribute matches
        the given `type_to_remove`. All associated particle properties are
        updated. Note: This resets the neighbor information, so
        update_neighbors should be called afterwards if neighbor data is
needed.
        :param type_to_remove: Particle type to remove.
        
        """
        # Create a mask for particles to keep
        mask = self.types != type_to_remove

        # Update particle count
        self.num_particles = np.count_nonzero(mask)

        # Update all particle properties using the mask
        self.position = self.position[mask]
        self.velocity = self.velocity[mask]
        self.viscosity_forces = self.viscosity_forces[mask]
        self.external_forces = self.external_forces[mask]
        self.pressure_forces = self.pressure_forces[mask]
        self.surface_tension_forces = self.surface_tension_forces[mask]
        self.mass = self.mass[mask]
        self.density = self.density[mask]
        self.density_intermediate = self.density_intermediate[mask]
        self.density_derivative = self.density_derivative[mask]
        self.alpha = self.alpha[mask]
        self.pressure = self.pressure[mask]
        self.types = self.types[mask]

        # Reset neighbor data (since indices may no longer be valid)
        self.neighbor_indices = np.zeros(0, dtype=np.int32)
        self.neighbor_counts = np.zeros(self.num_particles, dtype=np.int32)
        self.neighbor_starts = np.zeros(self.num_particles, dtype=np.int32)

    def __repr__(self):
        return f"Particles(num={self.num_particles})"