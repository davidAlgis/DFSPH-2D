import numpy as np
from dfsph.particle import Particle


def particles_init(grid_size, h, rest_density, spacing, box_origin, box_size):
    """
    Compute the particle layout based on grid size and initialize them inside a staggered box.
    
    :param grid_size: Tuple (grid_width, grid_height).
    :param num_particles: Total number of particles.
    :param mass: Mass of each particle.
    :param h: Smoothing length.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the box origin.
    :param box_size: (width, height) of the box.
    :return: List of initialized Particle instances.
    """
    # Compute max possible particles in the box
    num_particles_x = int(box_size[0] / spacing)
    num_particles_y = int(box_size[1] / spacing)
    num_particles = num_particles_x * num_particles_y

    # Compute mass dynamically based on box volume and rest density
    box_volume = box_size[0] * box_size[1]
    mass = (rest_density * box_volume) / num_particles

    return staggered_init(num_particles_x, num_particles_y, mass, h, spacing,
                          box_origin)


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
    :param num_particles: Total number of particles to create.
    :param mass: Mass of each particle.
    :param h: Smoothing length for SPH.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the bottom-left corner of the box.
    :return: List of Particle instances.
    """
    particles = []
    count = 0

    for i in range(num_particles_x):
        for j in range(num_particles_y):

            # Create staggered pattern by shifting every other row
            x_offset = (j % 2) * (spacing / 2)
            position = np.array([
                box_origin[0] + i * spacing + x_offset,
                box_origin[1] + j * spacing
            ])
            velocity = np.zeros(2)
            particles.append(Particle(count, position, velocity, mass, h))

            count += 1

    return particles
