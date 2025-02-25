import numpy as np
from dfsph.particle import Particle


def particles_init(grid_origin, grid_size, h, rest_density, spacing,
                   box_origin, box_size):
    """
    Compute the particle layout based on grid size and initialize them inside a staggered box.
    Additionally, adds solid static particles along the boundary of the box.
    
    :param grid_size: Tuple (grid_width, grid_height).
    :param h: Smoothing length.
    :param rest_density: Rest density of the fluid.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the box origin.
    :param box_size: (width, height) of the box.
    :return: List of initialized Particle instances.
    """
    # Compute max possible fluid particles in the box (staggered interior)
    num_particles_x = int(box_size[0] / spacing)
    num_particles_y = int(box_size[1] / spacing)
    num_fluid_particles = num_particles_x * num_particles_y

    # Compute mass dynamically based on box area and rest density.
    box_area = box_size[0] * box_size[1]
    # The magic number ensures the initial average density is close to rest_density.
    mass = (rest_density * box_area) / num_fluid_particles

    # Fluid particles using a staggered grid initialization.
    fluid_particles = staggered_init(num_particles_x, num_particles_y, mass, h,
                                     spacing, box_origin)

    # Boundary particles (solid static) along the grid boundary.
    boundary_particles = add_boundary_particles(len(fluid_particles),
                                                grid_origin, grid_size,
                                                spacing / 4, mass, h)

    # Combine fluid and boundary particles.
    all_particles = fluid_particles + boundary_particles

    return all_particles


def staggered_init(num_particles_x,
                   num_particles_y,
                   mass,
                   h,
                   spacing=1.0,
                   box_origin=(0.0, 0.0)):
    """
    Initialize fluid particles in a staggered grid pattern inside a box.
    
    :param num_particles_x: Number of fluid particles along the x-axis.
    :param num_particles_y: Number of fluid particles along the y-axis.
    :param mass: Mass of each fluid particle.
    :param h: Smoothing length for SPH.
    :param spacing: Distance between particles.
    :param box_origin: (x, y) coordinates of the bottom-left corner of the box.
    :return: List of fluid Particle instances.
    """
    particles = []
    count = 0

    for i in range(num_particles_x):
        for j in range(num_particles_y):
            # Create a staggered pattern by shifting every other row.
            x_offset = (j % 2) * (spacing / 2)
            position = np.array([
                box_origin[0] + i * spacing + x_offset,
                box_origin[1] + j * spacing
            ])
            velocity = np.zeros(2)
            # Fluid particles are marked as "fluid".
            particles.append(
                Particle(count,
                         position,
                         velocity,
                         mass,
                         h,
                         type_particle="fluid"))
            count += 1

    return particles


def add_boundary_particles(count, box_origin, box_size, spacing, mass, h):
    """
    Add solid static particles uniformly along the boundary of the simulation box.
    
    :param box_origin: (x, y) coordinates of the bottom-left corner of the box.
    :param box_size: (width, height) of the box.
    :param spacing: Spacing between boundary particles.
    :param mass: Mass for each boundary particle (can be same as fluid or adjusted).
    :param h: Smoothing length.
    :return: List of solid Particle instances.
    """
    boundary_particles = []

    x0, y0 = box_origin
    width, height = box_size
    x0 += h / 2
    y0 += h / 2
    x1 = x0 + width
    y1 = y0 + height
    x1 -= h
    y1 -= h

    # Bottom boundary (excluding corners if needed)
    x_positions = np.arange(x0, x1 + spacing, spacing)
    for x in x_positions:
        pos = np.array([x, y0])
        boundary_particles.append(
            Particle(count, pos, np.zeros(2), mass, h, type_particle="solid"))
        count += 1

    # Top boundary
    for x in x_positions:
        pos = np.array([x, y1])
        boundary_particles.append(
            Particle(count, pos, np.zeros(2), mass, h, type_particle="solid"))
        count += 1

    # Left boundary (excluding corners already added)
    y_positions = np.arange(y0 + spacing, y1, spacing)
    for y in y_positions:
        pos = np.array([x0, y])
        boundary_particles.append(
            Particle(count, pos, np.zeros(2), mass, h, type_particle="solid"))
        count += 1

    # Right boundary
    for y in y_positions:
        pos = np.array([x1, y])
        boundary_particles.append(
            Particle(count, pos, np.zeros(2), mass, h, type_particle="solid"))
        count += 1

    return []#boundary_particles
