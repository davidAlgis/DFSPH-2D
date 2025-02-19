from grid import Grid


def test_grid():
    # Define the grid parameters
    grid_size = (5, 5)  # 5x5 grid
    grid_position = (0.0, 0.0)  # Grid starts at (0, 0)
    cell_size = 1.0  # Each cell is 1x1 unit

    # Initialize the grid
    grid = Grid(grid_size, grid_position, cell_size)

    # Define some particles with positions
    particles = [
        {
            'position': [0.5, 0.5]
        },  # Particle in cell (0, 0)
        {
            'position': [1.5, 1.5]
        },  # Particle in cell (1, 1)
        {
            'position': [2.5, 2.5]
        },  # Particle in cell (2, 2)
        {
            'position': [3.5, 3.5]
        },  # Particle in cell (3, 3)
        {
            'position': [4.5, 4.5]
        },  # Particle in cell (4, 4)
    ]

    # Update the grid with particle positions
    grid.update_grid(particles)

    # Test finding neighbors for the particle at index 2 (position [2.5, 2.5])
    neighbors = grid.find_neighbors(2, particles)

    # Expected neighbors are particles in adjacent cells
    expected_neighbors_positions = [
        [1.5, 1.5],  # Cell (1, 1)
        [3.5, 3.5],  # Cell (3, 3)
    ]
    # Check if the found neighbors match the expected neighbors
    found_positions = [neighbor['position'] for neighbor in neighbors]
    assert all(pos in found_positions for pos in expected_neighbors_positions
               ), "Test failed: Neighbors do not match expected positions."

    print("Test passed: Neighbors match expected positions.")


if __name__ == "__main__":
    test_grid()
