import numpy as np
from grid import Grid

def test_grid():
    # Define grid parameters
    grid_size = (5, 5)  # 5x5 grid
    grid_position = (0.0, 0.0)  # Grid starts at (0, 0)
    cell_size = (1.0, 1.0)  # Each cell is 1x1 unit

    # Initialize the grid
    grid = Grid(grid_size, grid_position, cell_size)

    # Define some particles with positions
    particles = [
        {'position': np.array([0.5, 0.5])},  # Particle in cell (0, 0)
        {'position': np.array([1.5, 1.5])},  # Particle in cell (1, 1)
        {'position': np.array([2.5, 2.5])},  # Particle in cell (2, 2)
        {'position': np.array([3.5, 3.5])},  # Particle in cell (3, 3)
        {'position': np.array([4.5, 4.5])},  # Particle in cell (4, 4)
    ]

    # Update the grid with particle positions
    particles_cells, grid_cells_counts, particles_sorted = grid.update_grid(particles)

    # Test finding neighbors for the particle at index 0
    neighbors = grid.find_neighbors(0, particles, particles_sorted, grid_cells_counts)

    # Expected neighbors for particle at index 0 (cell (0, 0)) are particles in cells (0, 1), (1, 0), and (1, 1)
    expected_neighbors = [1]  # Only particle in cell (1, 1) is within the search range

    # Check if the found neighbors match the expected neighbors
    assert set(neighbors) == set(expected_neighbors), f"Test failed: expected {expected_neighbors}, but got {neighbors}"

    print("Test passed: Neighbors found correctly!")

if __name__ == "__main__":
    test_grid()
