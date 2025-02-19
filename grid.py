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
        return np.zeros(self.grid_size, dtype=int)

    def get_cell_index(self, position):
        """
        Get the cell index for a given position in the simulation space.

        :param position: Position in the simulation space (x, y).
        :return: Cell index as a tuple (i, j) for 2D.
        """
        return tuple(
            np.floor(
                (position - self.grid_position) / self.cell_size).astype(int))

    def fill_particle_cells_index(self, particles):
        """
        Fill the particle cells index based on their positions.

        :param particles: List of particles with their positions.
        :return: Array of cell indices for each particle.
        """
        particles_cells = np.array(
            [self.get_cell_index(p['position']) for p in particles])
        return particles_cells

    def fill_grid_cells_count(self, particles_cells):
        """
        Count the number of particles in each cell.

        :param particles_cells: Array of cell indices for each particle.
        :return: Array of particle counts for each cell.
        """
        grid_cells_counts = np.zeros(self.grid_size, dtype=int)
        for cell in particles_cells:
            grid_cells_counts[cell] += 1
        return grid_cells_counts

    def fill_particles_sorted(self, particles_cells, grid_cells_counts):
        """
        Sort particles based on their cell indices.

        :param particles_cells: Array of cell indices for each particle.
        :param grid_cells_counts: Array of particle counts for each cell.
        :return: Sorted array of particle indices.
        """
        particles_map = np.zeros_like(particles_cells, dtype=int)
        current_index = 0
        for cell in np.ndindex(self.grid_size):
            count = grid_cells_counts[cell]
            particles_map[particles_cells == cell] = np.arange(
                current_index, current_index + count)
            current_index += count
        return particles_map

    def find_neighbors(self, particle_index, particles, particles_sorted,
                       grid_cells_counts):
        """
        Find the neighbors of a given particle using the grid.

        :param particle_index: Index of the particle to find neighbors for.
        :param particles: List of particles with their positions.
        :param particles_sorted: Sorted array of particle indices.
        :param grid_cells_counts: Array of particle counts for each cell.
        :return: List of neighboring particle indices.
        """
        position = particles[particle_index]['position']
        cell_index = self.get_cell_index(position)
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_index[0] + dx, cell_index[1] + dy)
                if 0 <= neighbor_cell[0] < self.grid_size[
                        0] and 0 <= neighbor_cell[1] < self.grid_size[1]:
                    start_index = grid_cells_counts[neighbor_cell]
                    end_index = grid_cells_counts[
                        neighbor_cell] + grid_cells_counts[neighbor_cell]
                    for i in range(start_index, end_index):
                        neighbor_index = particles_sorted[i]
                        if neighbor_index != particle_index:
                            neighbors.append(neighbor_index)

        return neighbors

    def update_grid(self, particles):
        """
        Update the grid with the new particle positions.

        :param particles: List of particles with their positions.
        """
        particles_cells = self.fill_particle_cells_index(particles)
        grid_cells_counts = self.fill_grid_cells_count(particles_cells)
        particles_sorted = self.fill_particles_sorted(particles_cells,
                                                      grid_cells_counts)
        return particles_cells, grid_cells_counts, particles_sorted
