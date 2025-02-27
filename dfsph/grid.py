import numpy as np
import time
from collections import defaultdict
from scipy.spatial import cKDTree
from dfsph.particles import Particles


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
        """
        Returns the grid cell index (i, j) for a given position.
        
        Note: This method is kept for compatibility with the original design.
        """
        cell_idx = np.floor(
            (position - self.grid_origin) * self.inv_cell_size).astype(int)
        return tuple(cell_idx)

    def insert_particles(self, particles):
        """
        Inserts all particles into the grid.

        This method is maintained for legacy purposes. The accelerated neighbor search
        uses cKDTree instead.
        """
        self.cells.clear()  # Reset grid
        for i in range(particles.num_particles):
            cell_idx = self._compute_cell_index(particles.position[i])
            self.cells[cell_idx].append(i)

    def find_neighbors(self, particles, search_radius):
        """
        Finds neighbors for each particle using scipy's cKDTree for fast spatial queries.

        :param particles: The Particles object.
        :param search_radius: The radius within which to search for neighbors.
        """
        # Build the tree from particle positions
        tree = cKDTree(particles.position)
        # Query the tree: for each particle, find all indices within search_radius.
        all_neighbors = tree.query_ball_point(particles.position,
                                              search_radius)
        # Remove self from the neighbor list if present.
        particles.neighbors = []
        for i, neigh in enumerate(all_neighbors):
            filtered = [j for j in neigh if j != i]
            particles.neighbors.append(filtered)
