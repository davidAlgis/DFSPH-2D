import numpy as np


class Grid:

    def __init__(self, grid_size, grid_position, cell_size):
        """
        Initialize the grid with the given parameters.

        :param grid_size: Size of the grid (number of cells in each dimension).
        :param grid_position: Position of the grid in the simulation space.
        :param cell_size: Size of each cell in the grid.
        """
        self.grid_size = grid_size
        self.grid_position = grid_position
        self.cell_size = cell_size
        self.cells = self.initialize_cells()

    def initialize_cells(self):
        """
        Initialize the cells of the grid.
        """
        # Initialize an empty list for each cell
        return [[[] for _ in range(self.grid_size[1])]
                for _ in range(self.grid_size[0])]

    def update_grid(self, particles):
        """
        Update the grid with the new particle positions.

        :param particles: List of particles with their positions.
        """
        # Clear the current grid
        self.cells = self.initialize_cells()

        # Insert each particle into the corresponding cell
        for particle in particles:
            cell_index = self.get_cell_index(particle['position'])
            if self.is_within_bounds(cell_index):
                self.cells[cell_index[0]][cell_index[1]].append(particle)

    def find_neighbors(self, particle_index, particles):
        """
        Find the neighbors of a given particle using the grid.

        :param particle_index: Index of the particle to find neighbors for.
        :param particles: List of particles with their positions.
        :return: List of neighboring particle indices.
        """
        particle = particles[particle_index]
        cell_index = self.get_cell_index(particle['position'])
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
                    if neighbor != particle:
                        neighbors.append(neighbor)

        return neighbors

    def get_cell_index(self, position):
        """
        Get the cell index for a given position in the simulation space.

        :param position: Position in the simulation space.
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

        :param cell_index: Cell index as a tuple (i, j).
        :return: True if within bounds, False otherwise.
        """
        return 0 <= cell_index[0] < self.grid_size[0] and 0 <= cell_index[
            1] < self.grid_size[1]
