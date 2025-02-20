import numpy as np
from dfsph.particle import Particle


def particles_init(grid_size, num_particles, mass, h, spacing=1.0):

    # Compute grid layout for staggered initialization
    grid_width, grid_height = grid_size
    aspect_ratio = grid_width / grid_height  # Maintain proportional distribution
    num_particles_y = int(np.sqrt(num_particles / aspect_ratio))
    num_particles_x = int(num_particles / num_particles_y)
    return staggered_init(num_particles_x, num_particles_y, mass, h, spacing)


def staggered_init(num_particles_x,
                   num_particles_y,
                   mass,
                   h,
                   spacing=1.0,
                   box_origin=(0.0, 0.0)):
    """
    Initialize particles in a staggered grid pattern inside a box.

    :param num_particles_x: Number of particles along the x-axis.
    :param num_particles_y: Number of particles along the y-axis.
    :param mass: Mass of each particle.
    :param h: Smoothing length for SPH.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the bottom-left corner of the box.
    :return: List of Particle instances.
    """
    particles = []
    for i in range(num_particles_x):
        for j in range(num_particles_y):
            # Create staggered pattern by shifting every other row
            x_offset = (j % 2) * (spacing / 2)
            position = np.array([
                box_origin[0] + i * spacing + x_offset,
                box_origin[1] + j * spacing
            ])
            velocity = np.zeros(2)
            particles.append(Particle(position, velocity, mass, h))
    return particles
