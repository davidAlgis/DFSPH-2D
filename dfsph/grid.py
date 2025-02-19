import numpy as np
from dfsph.particle import Particle  # Import the Particle class


class Grid:

    def __init__(self, grid_size, grid_position, cell_size):
        """
        Initialize the grid with the given parameters.

        :param grid_size: Size of the grid (number of cells in each dimension).
        :param grid_position: Position of the grid in the simulation space.
        :param cell_size: Size of each cell in the grid.
        """
        self.grid_size = grid_size
        self.grid_position = np.array(grid_position, dtype=float)
        self.grid_end = self.grid_position + np.array(grid_size)
        self.cell_size = cell_size
        self.cells = self.initialize_cells()

    def initialize_cells(self):
        """
        Initialize the grid cells as a 2D list of empty lists.
        """
        return [[[] for _ in range(self.grid_size[1])]
                for _ in range(self.grid_size[0])]

    def update_grid(self, particles):
        """
        Update the grid by assigning particles to their respective cells.

        :param particles: List of Particle instances.
        """
        # Clear the grid before updating
        self.cells = self.initialize_cells()

        # Insert each particle into its corresponding cell
        for particle in particles:
            cell_index = self.get_cell_index(particle.position)
            if self.is_within_bounds(cell_index):
                self.cells[cell_index[0]][cell_index[1]].append(particle)

    def find_neighbors(self, particle):
        """
        Find the neighboring particles of a given particle using the grid.

        :param particle: The particle for which to find neighbors.
        :return: List of neighboring Particle instances.
        """
        cell_index = self.get_cell_index(particle.position)
        neighbors = []

        # Define the range of cells to search
        xmin, xmax = max(0, cell_index[0] - 1), min(self.grid_size[0],
                                                    cell_index[0] + 2)
        ymin, ymax = max(0, cell_index[1] - 1), min(self.grid_size[1],
                                                    cell_index[1] + 2)

        # Search in the neighboring cells
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                for neighbor in self.cells[x][y]:
                    if neighbor is not particle:  # Ensure it does not include itself
                        neighbors.append(neighbor)

        return neighbors

    def get_cell_index(self, position):
        """
        Get the grid cell index for a given particle position.

        :param position: Position as a numpy array [x, y].
        :return: Cell index as a tuple (i, j) for 2D.
        """
        return (int(
            np.floor((position[0] - self.grid_position[0]) / self.cell_size)),
                int(
                    np.floor((position[1] - self.grid_position[1]) /
                             self.cell_size)))

    def is_within_bounds(self, cell_index):
        """
        Check if a cell index is within the bounds of the grid.

        :param cell_index: Tuple (i, j) representing the cell index.
        :return: True if within bounds, False otherwise.
        """
        return 0 <= cell_index[0] < self.grid_size[0] and 0 <= cell_index[
            1] < self.grid_size[1]
