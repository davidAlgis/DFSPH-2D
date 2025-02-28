import numpy as np
from dfsph.particles import Particles


def particles_init(grid_origin, grid_size, h, rest_density, spacing,
                   box_origin, box_size):
    """
    Compute the particle layout based on the simulation domain and initialize them inside a staggered box.
    Additionally, adds solid static boundary particles along the edges of the simulation domain.
    
    :param grid_origin: (x, y) coordinates of the simulation domain's lower-left corner.
    :param grid_size: Tuple (grid_width, grid_height) defining the physical simulation area.
    :param h: Smoothing length.
    :param rest_density: Rest density of the fluid.
    :param spacing: Distance between fluid particles.
    :param box_origin: (x, y) coordinates of the fluid initialization box's origin.
    :param box_size: (width, height) of the fluid initialization box.
    :return: A Particles instance containing both fluid and boundary particles.
    """
    # Compute number of fluid particles to be placed in the box.
    num_particles_x = int(box_size[0] / spacing)
    num_particles_y = int(box_size[1] / spacing)
    num_fluid_particles = num_particles_x * num_particles_y

    # Compute mass dynamically so that the fluid's average density is close to rest_density.
    box_area = box_size[0] * box_size[1]
    mass = (rest_density * box_area) / num_fluid_particles

    # Create an initially empty Particles container.
    particles = Particles(num_particles=0)

    # Add fluid particles in a staggered grid pattern.
    particles = staggered_init(particles, num_particles_x, num_particles_y,
                               mass, h, spacing, box_origin)

    # Add boundary particles (solid static) along the simulation domain boundary.
    particles = add_boundary_particles(particles, grid_origin, grid_size,
                                       spacing, mass, h)

    return particles


def staggered_init(particles,
                   num_particles_x,
                   num_particles_y,
                   mass,
                   h,
                   spacing=1.0,
                   box_origin=(0.0, 0.0)):
    """
    Initialize fluid particles in a staggered grid pattern within a rectangular box.
    
    :param particles: Particles instance to add fluid particles to.
    :param num_particles_x: Number of fluid particles along the x-axis.
    :param num_particles_y: Number of fluid particles along the y-axis.
    :param mass: Mass of each fluid particle.
    :param h: Smoothing length for SPH.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the bottom-left corner of the fluid box.
    :return: Particles instance with fluid particles added (particle_type=0 for fluid).
    """
    for i in range(num_particles_x):
        for j in range(num_particles_y):
            # Create a staggered pattern by shifting every other row.
            x_offset = (j % 2) * (spacing / 2)
            position = (box_origin[0] + i * spacing + x_offset,
                        box_origin[1] + j * spacing)
            velocity = (0.0, 0.0)
            # Fluid particles are marked as "fluid" (particle_type 0).
            particles.add_particle(position,
                                   velocity,
                                   mass,
                                   alpha=0.0,
                                   particle_type=0)
    return particles


def add_boundary_particles(particles, box_origin, box_size, spacing, mass, h):
    """
    Add solid static boundary particles uniformly along the edges of the simulation domain.
    
    :param particles: Particles instance to add boundary particles to.
    :param box_origin: (x, y) coordinates of the simulation domain's lower-left corner.
    :param box_size: (width, height) of the simulation domain.
    :param spacing: Spacing between boundary particles.
    :param mass: Mass for each boundary particle.
    :param h: Smoothing length.
    :return: Particles instance with boundary particles added (particle_type=1 for solid).
    """
    # Adjust boundaries to avoid overlap with fluid region.
    x0, y0 = box_origin
    width, height = box_size
    x0 += h / 2
    y0 += h / 2
    x1 = x0 + width
    y1 = y0 + height
    x1 -= h
    y1 -= h

    # Bottom boundary.
    x_positions = np.arange(x0, x1 + spacing, spacing)
    for x in x_positions:
        pos = (x, y0)
        particles.add_particle(pos, (0.0, 0.0),
                               mass,
                               alpha=0.0,
                               particle_type=1)
    # Top boundary.
    for x in x_positions:
        pos = (x, y1)
        particles.add_particle(pos, (0.0, 0.0),
                               mass,
                               alpha=0.0,
                               particle_type=1)
    # Left boundary (excluding corners already added).
    y_positions = np.arange(y0 + spacing, y1, spacing)
    for y in y_positions:
        pos = (x0, y)
        particles.add_particle(pos, (0.0, 0.0),
                               mass,
                               alpha=0.0,
                               particle_type=1)
    # Right boundary.
    for y in y_positions:
        pos = (x1, y)
        particles.add_particle(pos, (0.0, 0.0),
                               mass,
                               alpha=0.0,
                               particle_type=1)

    return particles
