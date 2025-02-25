import numpy as np
from collections import defaultdict


class Grid:

    def __init__(self, grid_origin, grid_size, cell_size):
        """
        Initialize the grid with the given parameters.

        :param grid_origin: Position of the grid in the simulation space.
        :param grid_size: Physical dimensions of the grid (width, height).
        :param cell_size: Size of each cell in the grid.
        """
        self.grid_origin = np.array(grid_origin, dtype=float)
        self.grid_size = np.array(grid_size, dtype=float)
        self.grid_end = self.grid_origin + self.grid_size
        self.cell_size = cell_size

        self.inv_cell_size = 1.0 / cell_size  # Optimization for fast division
        self.cells = defaultdict(
            list)  # Hashmap {cell_idx: [particle_indices]}

    def _compute_cell_index(self, position):
        """ Returns the grid cell index (i, j) for a given position. """
        cell_idx = np.floor(
            (position - self.grid_origin) * self.inv_cell_size).astype(int)
        return tuple(cell_idx)

    def insert_particles(self, particles):
        """ Inserts all particles into the grid. """
        self.cells.clear()  # Reset grid
        for i in range(particles.num_particles):
            cell_idx = self._compute_cell_index(particles.position[i])
            self.cells[cell_idx].append(i)

    def find_neighbors(self, particles, search_radius):
        """
        Finds neighbors for each particle using the spatial grid.

        :param particles: The Particles object.
        :param search_radius: The radius within which to search for neighbors.
        """
        search_cells = int(np.ceil(search_radius /
                                   self.cell_size))  # Max cells to check
        particles.neighbors = [[] for _ in range(particles.num_particles)
                               ]  # Reset neighbors

        for i in range(particles.num_particles):
            cell_idx = self._compute_cell_index(particles.position[i])

            # Iterate over neighboring cells
            for dx in range(-search_cells, search_cells + 1):
                for dy in range(-search_cells, search_cells + 1):
                    neighbor_cell = (cell_idx[0] + dx, cell_idx[1] + dy)
                    if neighbor_cell in self.cells:
                        for j in self.cells[neighbor_cell]:
                            if i != j:
                                dist_sq = np.sum((particles.position[i] -
                                                  particles.position[j])**2)
                                if dist_sq < search_radius**2:
                                    particles.neighbors[i].append(j)
