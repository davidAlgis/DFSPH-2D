import numpy as np

# Global force identifiers
PRESSURE = "pressure"
VISCOSITY = "viscosity"
EXTERNAL = "external"
SURFACE_TENSION = "surface_tension"


class Particle:

    def __init__(self, position, velocity, mass, h):
        """
        Initialize a Particle object with its properties.

        :param position: Initial position as a 2D vector (list, tuple, or numpy array).
        :param velocity: Initial velocity as a 2D vector (list, tuple, or numpy array).
        :param mass: Mass of the particle.
        :param h: Support radius used in SPH calculations.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.h = h

        # SPH-related properties
        self.density = 0.0  # Computed density
        self.pressure = 0.0  # Computed pressure

        # Force dictionary for debugging each force separately
        self.forces = {
            PRESSURE: np.zeros(2, dtype=float),
            VISCOSITY: np.zeros(2, dtype=float),
            EXTERNAL:
            np.zeros(2, dtype=float),  # Gravity, external interactions, etc.
            SURFACE_TENSION: np.zeros(2, dtype=float)
        }

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
        Compute the total force acting on the particle by summing all forces.

        :return: A 2D vector representing the total accumulated force.
        """
        return sum(self.forces.values(), np.zeros(2, dtype=float))

    def debug_forces(self):
        """
        Print all forces applied to the particle for debugging purposes.
        """
        print("Forces acting on the particle:")
        for force_id, force_value in self.forces.items():
            print(f"  {force_id}: {force_value}")


# Example usage:
if __name__ == '__main__':
    # Create a particle at position (0,0) with initial velocity (1,0), mass 1, and support radius 1.0.
    particle = Particle(position=[0, 0], velocity=[1, 0], mass=1.0, h=1.0)

    # Apply different forces using global force IDs
    particle.add_force(EXTERNAL, [0, -9.81])  # Gravity
    particle.add_force(PRESSURE, [2, 3])  # Example pressure force
    particle.add_force(VISCOSITY, [-0.5, -0.2])  # Example viscosity force

    # Debug forces before updating
    particle.debug_forces()

    # Print updated state
    print("Position:", particle.position)
    print("Velocity:", particle.velocity)
