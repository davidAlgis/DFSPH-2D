import pytest
from dfsph.grid import Grid
from dfsph.particle import Particle


def test_grid():
    # Define the grid parameters
    grid_size = (5, 5)  # 5x5 grid
    grid_origin = (0.0, 0.0)  # Grid starts at (0.0, 0.0)
    cell_size = 1.0  # Each cell is 1x1 unit

    # Initialize the grid
    grid = Grid(grid_origin, grid_size, cell_size)

    # Define some particles with positions
    particles = [
        Particle(0, position=[0.5, 0.5], velocity=[0, 0], mass=1.0,
                 h=1.0),  # Cell (0, 0)
        Particle(1, position=[1.5, 1.5], velocity=[0, 0], mass=1.0,
                 h=1.0),  # Cell (1, 1)
        Particle(2, position=[2.5, 2.5], velocity=[0, 0], mass=1.0,
                 h=1.0),  # Cell (2, 2)
        Particle(3, position=[3.5, 3.5], velocity=[0, 0], mass=1.0,
                 h=1.0),  # Cell (3, 3)
        Particle(4, position=[4.5, 4.5], velocity=[0, 0], mass=1.0,
                 h=1.0),  # Cell (4, 4)
    ]

    # Update the grid with particles
    grid.update_grid(particles)

    # Test finding neighbors for the particle at index 2 (position [2.5, 2.5])
    neighbors = grid.find_neighbors(particles[2], h=1.0)

    # Expected neighbors are particles in adjacent cells
    expected_neighbors = [
        particles[1],  # Particle at [1.5, 1.5]
        particles[3],  # Particle at [3.5, 3.5]
    ]

    # Check if the found neighbors match the expected neighbors
    assert set(neighbors) == set(expected_neighbors), \
        f"Test failed: Neighbors do not match expected particles. Found: {neighbors}"
