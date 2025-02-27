import numpy as np
import pytest
from dfsph.particles import Particles
from dfsph.grid import Grid
from dfsph.sim import DFSPHSim


def test_find_neighbors_complex():
    # Create a 3x3 grid of particles with spacing 1.0
    spacing = 1.0
    grid_dim = 3
    num_particles = grid_dim**2
    particles = Particles(num_particles)

    # Set positions for a 3x3 grid: (0,0), (0,1), ..., (2,2)
    idx = 0
    for i in range(grid_dim):
        for j in range(grid_dim):
            particles.position[idx] = [i * spacing, j * spacing]
            idx += 1

    # Use a search radius h = 1.5 so that both adjacent and diagonal particles are neighbors.
    h = 1.5
    dt = 0.033
    # Define grid origin and grid size to fully cover our particle domain (with a margin)
    grid_origin = np.array([-0.5, -0.5], dtype=np.float32)
    grid_size = np.array([grid_dim * spacing, grid_dim * spacing],
                         dtype=np.float32) + 1.0
    cell_size = 1.0  # One cell per spacing unit

    # Create the simulation object.
    # Note: DFSPHSim.__init__ calls find_neighbors before particles are inserted in the grid.
    sim = DFSPHSim(particles, h, dt, grid_origin, grid_size, cell_size)

    # Populate the grid with our particles.
    sim.grid.insert_particles(particles)

    # Update the neighbor data structure using the grid's find_neighbors method.
    sim.find_neighbors()

    # Compute the expected neighbors for each particle using a brute-force approach.
    expected_neighbors = []
    for i in range(num_particles):
        pos_i = particles.position[i]
        neigh_set = set()
        for j in range(num_particles):
            if i == j:
                continue
            pos_j = particles.position[j]
            # Use the same condition as in grid.find_neighbors: strictly less than h^2
            if np.sum((pos_i - pos_j)**2) < h**2:
                neigh_set.add(j)
        expected_neighbors.append(neigh_set)

    # Validate that the neighbor data stored in the SoA format matches the expected neighbors.
    for i in range(num_particles):
        start = particles.neighbor_starts[i]
        count = particles.neighbor_counts[i]
        actual_neighbors = set(particles.neighbor_indices[start:start + count])
        assert actual_neighbors == expected_neighbors[i], (
            f"Particle {i}: expected neighbors {expected_neighbors[i]}, got {actual_neighbors}"
        )
