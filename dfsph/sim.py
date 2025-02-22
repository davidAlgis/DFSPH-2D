import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
import dfsph.sph_accelerated


class DFSPHSim:

    def __init__(self,
                 particles,
                 h,
                 dt,
                 grid_origin,
                 grid_size,
                 cell_size,
                 rest_density=1027,
                 water_viscosity=0.1):
        """
        Initialize the DFSPH simulation.
        """
        self.particles = particles
        self.num_particles = len(self.particles)
        self.h = h
        self.dt = dt  # Initial dt; will be adapted dynamically.
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.mean_density = 0

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_origin, grid_size, cell_size)

        # Physical parameters
        self.gravity = np.array([0, -9.81])  # Gravity force

        # Buffers for preallocation (will be allocated in preallocate_buffers())
        self.positions_buf = None
        self.velocities_buf = None
        self.masses_buf = None
        self.neighbor_counts_buf = None
        self.neighbor_starts_buf = None
        self.neighbor_indices_buf = None

        self.preallocate_buffers()

    def preallocate_buffers(self):
        """
        Preallocate arrays for particle data to minimize dynamic allocations.
        Assumes an estimated maximum of 20 neighbors per particle.
        """
        N = self.num_particles
        nbr_neighbor_max = int(N / 5)
        self.positions_buf = np.empty((N, 2), dtype=np.float64)
        self.velocities_buf = np.empty((N, 2), dtype=np.float64)
        self.masses_buf = np.empty(N, dtype=np.float64)
        self.neighbor_counts_buf = np.empty(N, dtype=np.int32)
        self.neighbor_starts_buf = np.empty(N, dtype=np.int32)
        # Estimate maximum neighbor indices buffer size (adjust as needed)
        self.neighbor_indices_buf = np.empty(nbr_neighbor_max * N,
                                             dtype=np.int32)

    def _pack_particle_data(self):
        """
        Pack particle data into preallocated arrays.
        Reuses buffers allocated in preallocate_buffers() to avoid per-step allocations.
        Returns:
            positions: (N,2) array of particle positions.
            velocities: (N,2) array of particle velocities.
            masses: (N,) array of particle masses.
            neighbor_indices: 1D array of neighbor indices (sized to actual count).
            neighbor_starts: (N,) array indicating start indices in neighbor_indices for each particle.
            neighbor_counts: (N,) array with the number of neighbors for each particle.
        """
        N = self.num_particles
        pos = self.positions_buf
        vel = self.velocities_buf
        mass = self.masses_buf
        n_counts = self.neighbor_counts_buf
        n_starts = self.neighbor_starts_buf
        neighbor_list = self.neighbor_indices_buf

        index_ptr = 0
        for i, particle in enumerate(self.particles):
            pos[i, :] = particle.position
            vel[i, :] = particle.velocity
            mass[i] = particle.mass
            n_neighbors = len(particle.neighbors)
            n_counts[i] = n_neighbors
            n_starts[i] = index_ptr

            # Fill the neighbor indices for particle i
            for neighbor in particle.neighbors:
                if index_ptr < neighbor_list.shape[0]:
                    neighbor_list[index_ptr] = neighbor.index
                else:
                    # Optionally, reallocate a larger buffer here.
                    raise ValueError(
                        "Exceeded preallocated neighbor indices buffer size")
                index_ptr += 1

        # Return only the used portion of the neighbor indices array.
        neighbor_indices = neighbor_list[:index_ptr].copy()
        return pos, vel, mass, neighbor_indices, n_starts, n_counts

    def compute_density_and_alpha(self):
        """
        Compute the density and alpha coefficient for each particle using Numba.
        """
        positions, _, masses, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data(
        )

        densities, alphas = dfsph.sph_accelerated.compute_density_alpha_numba(
            positions, masses, neighbor_indices, neighbor_starts,
            neighbor_counts, self.h, self.rest_density)

        total_density = 0.0
        for i, particle in enumerate(self.particles):
            particle.density = densities[i]
            particle.alpha = alphas[i]
            total_density += densities[i]

        self.mean_density = total_density / self.num_particles

    def compute_pressure(self, B=1000.0, gamma=7):
        """
        Compute the pressure for each particle using Tait's equation:
            p = B * ((density / rest_density)**gamma - 1)
        """
        for particle in self.particles:
            particle.pressure = B * (
                (particle.density / self.rest_density)**gamma - 1)

    def compute_pressure_forces_wcsph(self):
        """
        Compute the pressure forces using Numba acceleration.
        Pressure force formulation:
            F_pressure = - sum_j m_j * (p_i/(rho_i^2) + p_j/(rho_j^2)) * gradW_ij
        """
        positions, velocities, masses, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data(
        )
        pressures = np.array([p.pressure for p in self.particles],
                             dtype=np.float64)
        densities = self.get_particle_densities()

        pressure_forces = dfsph.sph_accelerated.compute_pressure_forces_wcsph(
            positions, pressures, densities, masses, neighbor_indices,
            neighbor_starts, neighbor_counts, self.h)

        for i, particle in enumerate(self.particles):
            particle.add_force(PRESSURE, pressure_forces[i])

    def compute_viscosity_forces(self):
        """
        Compute the viscosity forces using Numba.
        """
        positions, velocities, masses, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data(
        )

        viscosity_forces = dfsph.sph_accelerated.compute_viscosity_forces(
            positions, velocities, self.get_particle_densities(), masses,
            neighbor_indices, neighbor_starts, neighbor_counts, self.h,
            self.water_viscosity)

        for i, particle in enumerate(self.particles):
            particle.add_force(VISCOSITY, viscosity_forces[i])

    def get_particle_densities(self):
        """
        Return an array of densities for the particles.
        """
        return np.array([p.density for p in self.particles], dtype=np.float64)

    def predict_intermediate_velocity(self):
        """
        Apply external forces (e.g., gravity) to update particle velocities.
        """
        for particle in self.particles:
            particle.add_force(EXTERNAL, particle.mass * self.gravity)
            total_force = particle.total_force()
            acceleration = total_force / particle.mass
            particle.velocity += acceleration * self.dt

    def integrate(self):
        """
        Integrate particle velocities and positions using Euler integration.
        """
        for particle in self.particles:
            particle.position += particle.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        """
        Apply penalty forces when particles hit the simulation boundaries.
        """
        for particle in self.particles:
            for i in range(2):  # For 2D: x (0) and y (1)
                if particle.position[i] < self.grid.grid_origin[i] + 1e-2:
                    particle.position[
                        i] = self.grid.grid_origin[i] + self.h * (
                            self.grid.grid_origin[i] - particle.position[i])
                    particle.velocity[i] *= -collider_damping
                elif particle.position[i] > self.grid.grid_origin[
                        i] + self.grid.grid_size[i] - 1e-2:
                    particle.position[i] = (
                        self.grid.grid_origin[i] + self.grid.grid_size[i]
                    ) - self.h * (
                        particle.position[i] -
                        (self.grid.grid_origin[i] + self.grid.grid_size[i]))
                    particle.velocity[i] *= -collider_damping

    def find_neighbors(self):
        """
        Update each particle's neighbor list using the grid.
        """
        self.grid.update_grid(self.particles)
        for particle in self.particles:
            particle.neighbors = self.grid.find_neighbors(particle)

    def adapt_dt_for_cfl(self):
        """
        Adapt the time step (dt) dynamically based on the CFL condition.
        Uses:
            dt = 0.3999 * h / velocity_max
        and clamps dt between 0.0001 and 0.002.
        """
        velocity_max = 0.0
        for particle in self.particles:
            vel_norm = np.sqrt(particle.velocity[0]**2 +
                               particle.velocity[1]**2)
            if vel_norm > velocity_max:
                velocity_max = vel_norm

        if velocity_max < 1e-6:
            new_dt = 0.033
        else:
            new_dt = 0.3999 * self.h / velocity_max

        self.dt = max(0.0001, min(new_dt, 0.033))

    def update(self):
        """
        Perform a DFSPH update step:
            - Adapt dt based on CFL condition.
            - Reset forces.
            - Find neighbors.
            - Compute density, alpha, and pressure.
            - Compute pressure and viscosity forces.
            - Apply external forces.
            - Integrate velocities and positions.
            - Enforce boundary conditions.
        """
        # Adapt dt dynamically based on current particle velocities.
        self.adapt_dt_for_cfl()

        for particle in self.particles:
            particle.reset_forces()

        self.find_neighbors()
        self.compute_density_and_alpha()
        self.compute_pressure()
        self.compute_pressure_forces_wcsph()
        self.compute_viscosity_forces()
        self.predict_intermediate_velocity()
        self.integrate()
        self.apply_boundary_penalty()
