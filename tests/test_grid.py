import numpy as np
import pytest
from dfsph.grid import Grid
from dfsph.particles import Particles


def test_compute_cell_index():
    """
    Test the _compute_cell_index method of the Grid class.

    This test verifies that given a position, the correct cell index is computed
    based on the grid origin and cell size.
    """
    grid_origin = (10, 10)
    grid_size = (20, 20)
    cell_size = 2.0  # Cell size different from 1
    grid = Grid(grid_origin, grid_size, cell_size)

    # Test a position exactly one cell away from the grid origin.
    pos = np.array([12, 12])
    cell_idx = grid._compute_cell_index(pos)
    # (12-10)/2 = 1, so expected index is (1, 1)
    assert cell_idx == (1, 1)

    # Test a position exactly at the grid origin.
    pos_origin = np.array([10, 10])
    cell_idx_origin = grid._compute_cell_index(pos_origin)
    assert cell_idx_origin == (0, 0)


def test_insert_particles():
    """
    Test that particles are correctly inserted into the grid cells.

    This function adds particles at specific positions and checks that they are
    stored in the expected cells in the grid.
    """
    grid_origin = (10, 10)
    grid_size = (20, 20)
    cell_size = 2.0
    grid = Grid(grid_origin, grid_size, cell_size)

    # Create an empty Particles object and add two particles.
    particles = Particles(num_particles=0)
    # Particle 0 at (12,12) should go into cell (1,1)
    particles.add_particle((12, 12))
    # Particle 1 at (15,15): (15-10)/2 = 2.5 -> floor to 2 so cell (2,2)
    particles.add_particle((15, 15))

    grid.insert_particles(particles)

    # Check that each expected cell contains the correct particle indices.
    assert (1, 1) in grid.cells
    assert (2, 2) in grid.cells
    assert 0 in grid.cells[(1, 1)]
    assert 1 in grid.cells[(2, 2)]


def test_find_neighbors():
    """
    Test the neighbor search functionality of the Grid class.

    This function adds three particles, with two close enough to be neighbors
    and one far away, and verifies that neighbor relationships are correctly detected.
    """
    grid_origin = (10, 10)
    grid_size = (20, 20)
    cell_size = 2.0
    grid = Grid(grid_origin, grid_size, cell_size)

    particles = Particles(num_particles=0)
    # Add three particles:
    # Particle 0 and Particle 1 are close to each other.
    particles.add_particle((12, 12))  # Particle 0
    particles.add_particle((12.5, 12.5))  # Particle 1
    # Particle 2 is placed far from the others.
    particles.add_particle((18, 18))  # Particle 2

    grid.insert_particles(particles)

    # Define a search radius that captures the distance between particles 0 and 1.
    search_radius = 2.0
    grid.find_neighbors(particles, search_radius)

    # Verify that:
    # - Particle 0 detects Particle 1 as a neighbor.
    # - Particle 1 detects Particle 0 as a neighbor.
    # - Particle 2 should have no neighbors within the search radius.
    assert 1 in particles.neighbors[0]
    assert 0 in particles.neighbors[1]
    assert particles.neighbors[2] == []


def test_grid_end():
    """
    Test that the grid_end attribute is correctly computed.

    The grid_end should equal grid_origin plus grid_size.
    """
    grid_origin = (10, 10)
    grid_size = (20, 20)
    cell_size = 2.0
    grid = Grid(grid_origin, grid_size, cell_size)

    expected_end = np.array(grid_origin) + np.array(grid_size, dtype=float)
    assert np.allclose(grid.grid_end, expected_end)


def test_many_particles():
    """
    Test the grid insertion and neighbor search with many particles.

    This test adds a larger number of particles with random positions within the grid bounds,
    performs insertion and neighbor search, and then checks that every neighbor pair is within
    the specified search radius.
    """
    grid_origin = (5, 5)
    grid_size = (30, 30)
    cell_size = 1.5  # Non-default cell size
    grid = Grid(grid_origin, grid_size, cell_size)

    # Use a fixed seed for reproducibility.
    np.random.seed(42)
    num_particles = 100
    particles = Particles(num_particles=0)

    # Generate random positions within the grid bounds.
    positions = np.random.uniform(
        low=grid_origin[0],
        high=grid_origin[0] + grid_size[0],
        size=(num_particles, 2),
    )

    # Add particles to the system.
    for pos in positions:
        particles.add_particle(tuple(pos))

    grid.insert_particles(particles)

    # Set the search radius to the cell size.
    search_radius = 1.5
    grid.find_neighbors(particles, search_radius)

    # For each particle, verify that each neighbor is within the search radius.
    for i in range(particles.num_particles):
        pos_i = particles.position[i]
        for j in particles.neighbors[i]:
            pos_j = particles.position[j]
            dist = np.linalg.norm(pos_i - pos_j)
            assert (
                dist < search_radius
            ), f"Neighbor distance {dist:.2f} >= search radius for particles {i} and {j}"
