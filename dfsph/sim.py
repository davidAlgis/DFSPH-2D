import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
import dfsph.sph_accelerated
from dfsph.kernels import w, grad_w  # SPH kernel and its gradient


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
        # Gamma used in solid mass update
        self.gamma_mass_solid = 1.4

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_origin, grid_size, cell_size)

        # Physical parameters
        self.gravity = np.array([0, -9.81])  # Gravity force

        # Buffers for preallocation (allocated in preallocate_buffers())
        self.positions_buf = None
        self.velocities_buf = None
        self.masses_buf = None
        self.neighbor_counts_buf = None
        self.neighbor_starts_buf = None
        self.neighbor_indices_buf = None

        self.preallocate_buffers()
        # For solid neighbors we may wish to use a specific viscosity coefficient.
        self.viscosity_coefficient_solid = 0.01

    def preallocate_buffers(self):
        """
        Preallocate arrays for particle data to minimize dynamic allocations.
        Assumes an estimated maximum of (N/5) neighbors per particle.
        """
        N = self.num_particles
        nbr_neighbor_max = int(N / 5)
        self.positions_buf = np.empty((N, 2), dtype=np.float64)
        self.velocities_buf = np.empty((N, 2), dtype=np.float64)
        self.masses_buf = np.empty(N, dtype=np.float64)
        self.neighbor_counts_buf = np.empty(N, dtype=np.int32)
        self.neighbor_starts_buf = np.empty(N, dtype=np.int32)
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
            for neighbor in particle.neighbors:
                if index_ptr < neighbor_list.shape[0]:
                    neighbor_list[index_ptr] = neighbor.index
                else:
                    raise ValueError(
                        "Exceeded preallocated neighbor indices buffer size")
                index_ptr += 1
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

    def update_mass_solid(self):
        """
        Update the mass of solid particles based on their neighboring solid particles.
        For each solid particle, compute:
            massSolid = sum( w(p_i, p_j, h) ) for each neighbor j (only if j is solid and j != i)
            Then set: particle.mass = rest_density * gamma_mass_solid / massSolid
        """
        for particle in self.particles:
            if particle.type_particle == "solid":
                massSolid = 0.0
                for neighbor in particle.neighbors:
                    if neighbor.index != particle.index and neighbor.type_particle == "solid":
                        massSolid += w(particle.position, neighbor.position,
                                       self.h)
                if massSolid > 1e-8:
                    particle.mass = self.rest_density * self.gamma_mass_solid / massSolid

    def compute_viscosity_forces_updated(self):
        """
        Compute viscosity forces on fluid particles, taking into account both fluid and solid neighbors.
        For each fluid particle:
          - If neighbor is fluid:
              viscosity contribution = (mass/density_i) * waterViscosity * mass * (v_dot_r/denom) * grad_wij
              where denom = ||r_ij||^2 + 0.01 * h^2 * density_neighbor.
          - If neighbor is solid:
              viscosity contribution = (mass/density_i) * viscosity_coefficient_solid * (neighbor.mass) * (v_dot_r/denom) * grad_wij
              where denom = ||r_ij||^2 + 0.01 * h^2 * density_i.
        Finally, the total viscosity force is multiplied by 10.0.
        """
        for particle in self.particles:
            if particle.type_particle != "fluid":
                continue
            viscosity_force = np.zeros(2, dtype=np.float64)
            for neighbor in particle.neighbors:
                if neighbor.index == particle.index:
                    continue
                r_ij = particle.position - neighbor.position
                r2 = r_ij[0]**2 + r_ij[1]**2
                if r2 > self.h**2:
                    continue  # Outside kernel support.
                v_ij = particle.velocity - neighbor.velocity
                v_dot_r = v_ij[0] * r_ij[0] + v_ij[1] * r_ij[1]
                if neighbor.type_particle == "fluid":
                    denom = r2 + 0.01 * self.h**2 * neighbor.density
                    grad = grad_w(particle.position, neighbor.position, self.h)
                    contribution = (
                        particle.mass / particle.density
                    ) * self.water_viscosity * particle.mass * (v_dot_r /
                                                                denom) * grad
                    viscosity_force += contribution
                elif neighbor.type_particle == "solid":
                    # Use a separate viscosity coefficient for solids.
                    coeff = self.viscosity_coefficient_solid
                    denom = r2 + 0.01 * self.h**2 * particle.density
                    grad = grad_w(particle.position, neighbor.position, self.h)
                    contribution = (particle.mass / particle.density
                                    ) * coeff * neighbor.mass * (v_dot_r /
                                                                 denom) * grad
                    viscosity_force += contribution
            particle.add_force(VISCOSITY, 10.0 * viscosity_force)

    def compute_pressure_forces_updated(self):
        """
        Compute pressure forces on fluid particles, taking into account both fluid and solid neighbors.
        For each fluid particle:
          - If neighbor is fluid:
              pressure contribution = - mass * mass * pressure_neighbor / (density_i * density_neighbor) * grad_wij.
          - If neighbor is solid:
              pressure contribution = - mass * neighbor.mass * (pressure_i / density_i^2) * grad_wij.
        """
        for particle in self.particles:
            if particle.type_particle != "fluid":
                continue
            pressure_force = np.zeros(2, dtype=np.float64)
            density_i = particle.density
            density_i_sq = density_i * density_i
            pressure_i = particle.pressure
            for neighbor in particle.neighbors:
                if neighbor.index == particle.index:
                    continue
                grad = grad_w(particle.position, neighbor.position, self.h)
                if neighbor.type_particle == "fluid":
                    pressure_force += -particle.mass * particle.mass * neighbor.pressure / (
                        density_i * neighbor.density) * grad
                elif neighbor.type_particle == "solid":
                    pressure_force += -particle.mass * neighbor.mass * (
                        pressure_i / density_i_sq) * grad
            particle.add_force(PRESSURE, pressure_force)

    def predict_intermediate_velocity(self):
        """
        Apply external forces (e.g., gravity) to update particle velocities.
        Only fluid particles are updated; solid particles remain static.
        """
        for particle in self.particles:
            if particle.type_particle == "fluid":
                particle.add_force(EXTERNAL, particle.mass * self.gravity)
                total_force = particle.total_force()
                acceleration = total_force / particle.mass
                particle.velocity += acceleration * self.dt

    def integrate(self):
        """
        Integrate particle velocities and positions using Euler integration.
        Only fluid particles are integrated; solid particles remain static.
        """
        for particle in self.particles:
            if particle.type_particle == "fluid":
                particle.position += particle.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        """
        Apply penalty forces when fluid particles hit the simulation boundaries.
        Solid particles are not moved.
        """
        for particle in self.particles:
            if particle.type_particle == "fluid":
                for i in range(2):  # For 2D: x (0) and y (1)
                    if particle.position[i] < self.grid.grid_origin[i] + 1e-2:
                        particle.position[i] = self.grid.grid_origin[
                            i] + self.h * (self.grid.grid_origin[i] -
                                           particle.position[i])
                        particle.velocity[i] *= -collider_damping
                    elif particle.position[i] > self.grid.grid_origin[
                            i] + self.grid.grid_size[i] - 1e-2:
                        particle.position[i] = (
                            self.grid.grid_origin[i] + self.grid.grid_size[i]
                        ) - self.h * (particle.position[i] -
                                      (self.grid.grid_origin[i] +
                                       self.grid.grid_size[i]))
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
        and clamps dt between 0.0001 and 0.033.
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
            - Update mass for solid particles.
            - Compute density, alpha, and pressure.
            - Compute viscosity and pressure forces (including solid contributions).
            - Apply external forces.
            - Integrate velocities and positions.
            - Enforce boundary conditions.
        """
        self.adapt_dt_for_cfl()

        for particle in self.particles:
            particle.reset_forces()

        self.find_neighbors()
        self.update_mass_solid()
        self.compute_density_and_alpha()
        self.compute_pressure()
        self.compute_viscosity_forces_updated()
        self.compute_pressure_forces_updated()
        self.predict_intermediate_velocity()
        self.integrate()
        # self.apply_boundary_penalty()
