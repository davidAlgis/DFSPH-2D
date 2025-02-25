import unittest
import numpy as np
from dfsph.particle import Particle, PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION


class TestParticle(unittest.TestCase):

    def setUp(self):
        # Create a basic fluid particle.
        self.index = 0
        self.position = [1.0, 2.0]
        self.velocity = [0.5, -0.5]
        self.mass = 1.0
        self.h = 0.1
        self.type_particle = "fluid"
        self.particle = Particle(self.index, self.position, self.velocity,
                                 self.mass, self.h, self.type_particle)

    def test_initialization(self):
        # Test that attributes are correctly set.
        self.assertEqual(self.particle.index, self.index)
        np.testing.assert_array_almost_equal(self.particle.position,
                                             np.array(self.position))
        np.testing.assert_array_almost_equal(self.particle.velocity,
                                             np.array(self.velocity))
        self.assertEqual(self.particle.mass, self.mass)
        self.assertEqual(self.particle.h, self.h)
        self.assertEqual(self.particle.type_particle, self.type_particle)
        self.assertEqual(self.particle.density, 0.0)
        self.assertEqual(self.particle.alpha, 0.0)
        self.assertEqual(self.particle.pressure, 0.0)
        # Check forces are zeroed
        for force_id in [PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION]:
            np.testing.assert_array_almost_equal(
                self.particle.forces[force_id], np.zeros(2))
        self.assertEqual(self.particle.neighbors, [])

    def test_reset_forces(self):
        # Set non-zero forces then reset.
        self.particle.forces[PRESSURE] = np.array([1.0, 2.0])
        self.particle.forces[VISCOSITY] = np.array([3.0, 4.0])
        self.particle.reset_forces()
        for force_id in [PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION]:
            np.testing.assert_array_almost_equal(
                self.particle.forces[force_id], np.zeros(2))

    def test_add_force(self):
        # Reset and then add forces.
        self.particle.reset_forces()
        self.particle.add_force(PRESSURE, [1.0, 1.0])
        self.particle.add_force(VISCOSITY, [2.0, 2.0])
        np.testing.assert_array_almost_equal(self.particle.forces[PRESSURE],
                                             np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(self.particle.forces[VISCOSITY],
                                             np.array([2.0, 2.0]))
        # Check total force (other forces are zero)
        total = self.particle.total_force()
        np.testing.assert_array_almost_equal(total, np.array([3.0, 3.0]))

    def test_set_neighbors(self):
        # Create a neighbor particle and set it.
        neighbor = Particle(1, [0.0, 0.0], [0, 0], 1.0, 0.1)
        self.particle.set_neighbors([neighbor])
        self.assertEqual(len(self.particle.neighbors), 1)
        self.assertEqual(self.particle.neighbors[0].index, neighbor.index)

    def test_total_force(self):
        # Verify that total_force sums all individual forces correctly.
        self.particle.reset_forces()
        forces_to_add = {
            PRESSURE: [1.0, 0.0],
            VISCOSITY: [0.0, 2.0],
            EXTERNAL: [3.0, 0.0],
            SURFACE_TENSION: [0.0, 4.0]
        }
        for key, force in forces_to_add.items():
            self.particle.add_force(key, force)
        expected_total = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(self.particle.total_force(),
                                             expected_total)
