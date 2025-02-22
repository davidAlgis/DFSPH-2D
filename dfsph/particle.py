import numpy as np

# Global force identifiers
PRESSURE = "pressure"
VISCOSITY = "viscosity"
EXTERNAL = "external"
SURFACE_TENSION = "surface_tension"


class Particle:

    def __init__(self,
                 index,
                 position,
                 velocity,
                 mass,
                 h,
                 type_particle="fluid"):
        """
        Initialize a Particle object with its properties.

        :param position: Initial position as a 2D vector.
        :param velocity: Initial velocity as a 2D vector.
        :param mass: Mass of the particle.
        :param h: Support radius used in SPH calculations.
        :param type_particle: Either "fluid" or "solid".
        """
        self.index = index
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.h = h
        self.type_particle = type_particle  # "fluid" or "solid"

        # SPH-related properties
        self.density = 0.0
        self.alpha = 0.0
        self.pressure = 0.0

        # Force dictionary for debugging each force separately
        self.forces = {
            PRESSURE: np.zeros(2, dtype=float),
            VISCOSITY: np.zeros(2, dtype=float),
            EXTERNAL:
            np.zeros(2, dtype=float),  # Gravity, external interactions, etc.
            SURFACE_TENSION: np.zeros(2, dtype=float)
        }

        # Stores the list of neighbors for optimization
        self.neighbors = []

    def reset_forces(self):
        """
        Reset all forces to zero before the next simulation step.
        """
        for key in self.forces:
            self.forces[key].fill(0.0)

    def add_force(self, force_id, force_value):
        """
        Add a force to the particle, categorized by type for debugging.

        :param force_id: A string identifier (e.g., PRESSURE, VISCOSITY).
        :param force_value: A 2D vector representing the force to add.
        """
        if force_id in self.forces:
            self.forces[force_id] += np.array(force_value, dtype=float)
        else:
            raise ValueError(f"Unknown force type: {force_id}")

    def total_force(self):
        """
        Compute the total force acting on the particle.
        :return: A 2D vector representing the total accumulated force.
        """
        return sum(self.forces.values(), np.zeros(2, dtype=float))

    def set_neighbors(self, neighbors):
        """
        Store the list of neighboring particles.
        :param neighbors: List of neighboring Particle instances.
        """
        self.neighbors = neighbors

    def debug_forces(self):
        """
        Print all forces applied to the particle for debugging.
        """
        print("Forces acting on the particle:")
        for force_id, force_value in self.forces.items():
            print(f"  {force_id}: {force_value}")
