import numpy as np
from collections import defaultdict
from numba import njit
import math


@njit
def compute_cell_indices_numba(positions, grid_origin, cell_size):
    """
    For each particle, compute its cell index based on the grid origin and cell size.
    Returns an (n, 2) int64 array of cell indices.
    """
    n = positions.shape[0]
    cell_indices = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        cell_indices[i, 0] = int(
            math.floor((positions[i, 0] - grid_origin[0]) / cell_size))
        cell_indices[i, 1] = int(
            math.floor((positions[i, 1] - grid_origin[1]) / cell_size))
    return cell_indices


@njit
def find_neighbors_numba(positions, cell_indices, search_radius, cell_size):
    """
    Computes the neighbors for each particle using a grid-based approach.
    Returns:
      - neighbor_indices: a flat array of neighbor indices.
      - neighbor_counts: number of neighbors for each particle.
      - neighbor_starts: starting index in neighbor_indices for each particle.
    """
    n = positions.shape[0]
    search_radius_sq = search_radius * search_radius
    search_cells = int(math.ceil(search_radius / cell_size))

    neighbor_counts = np.empty(n, dtype=np.int64)
    neighbor_starts = np.empty(n, dtype=np.int64)
    total = 0

    # First pass: count neighbors for each particle.
    for i in range(n):
        count = 0
        cell_i0 = cell_indices[i, 0]
        cell_i1 = cell_indices[i, 1]
        for j in range(n):
            if i == j:
                continue
            if (abs(cell_indices[j, 0] - cell_i0) <= search_cells
                    and abs(cell_indices[j, 1] - cell_i1) <= search_cells):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                if dx * dx + dy * dy < search_radius_sq:
                    count += 1
        neighbor_counts[i] = count
        neighbor_starts[i] = total
        total += count

    neighbor_indices = np.empty(total, dtype=np.int64)

    # Second pass: fill in neighbor indices.
    for i in range(n):
        idx = neighbor_starts[i]
        cell_i0 = cell_indices[i, 0]
        cell_i1 = cell_indices[i, 1]
        for j in range(n):
            if i == j:
                continue
            if (abs(cell_indices[j, 0] - cell_i0) <= search_cells
                    and abs(cell_indices[j, 1] - cell_i1) <= search_cells):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                if dx * dx + dy * dy < search_radius_sq:
                    neighbor_indices[idx] = j
                    idx += 1
    return neighbor_indices, neighbor_counts, neighbor_starts


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
        self.inv_cell_size = 1.0 / cell_size

        # Maintained for the non-Numba neighbor search.
        self.cells = defaultdict(list)

    def _compute_cell_index(self, position):
        """
        Returns the grid cell index (i, j) for a given position.
        """
        cell_idx = np.floor(
            (position - self.grid_origin) * self.inv_cell_size).astype(int)
        return tuple(cell_idx)

    def insert_particles(self, particles):
        """
        Inserts all particles into the grid's cell dictionary.
        This method is used for the pure-Python neighbor search.
        """
        self.cells.clear()  # Reset grid
        for i in range(particles.num_particles):
            cell_idx = self._compute_cell_index(particles.position[i])
            self.cells[cell_idx].append(i)

    def find_neighbors(self, particles, search_radius):
        """
        Finds neighbors for each particle using the Numba-accelerated approach.
        This function computes the flat neighbor arrays and then reconstructs
        a list-of-lists structure (stored in particles.neighbors) for backward compatibility.
        
        :param particles: A Particles object with attribute 'position' (NumPy array).
        :param search_radius: The radius within which to search for neighbors.
        :return: A list-of-lists containing neighbor indices for each particle.
        """
        # Ensure positions are in float64 for Numba.
        positions = particles.position.astype(np.float64)
        cell_indices = compute_cell_indices_numba(positions, self.grid_origin,
                                                  self.cell_size)
        neighbor_indices, neighbor_counts, neighbor_starts = find_neighbors_numba(
            positions, cell_indices, search_radius, self.cell_size)

        # Build the list-of-lists structure.
        n = particles.num_particles
        neighbors_list = []
        for i in range(n):
            start = neighbor_starts[i]
            count = neighbor_counts[i]
            neighbors_list.append(list(neighbor_indices[start:start + count]))

        # Update the Particles object with both representations.
        particles.neighbors = neighbors_list
        particles.neighbor_indices = neighbor_indices
        particles.neighbor_counts = neighbor_counts
        particles.neighbor_starts = neighbor_starts

        return neighbors_list
