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
        # Viscosity coefficient for interactions with solids
        self.viscosity_coefficient_solid = 1.0

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
        (Not used when the constant density solver is active.)
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

    # --- Constant Density Pressure Solver Methods ---

    def update_density_intermediate(self, dt_sph):
        """
        For each fluid particle, compute an intermediate density value.
        The correction is based on velocity differences with neighbors.
        Fluid neighbors contribute with the particle mass; solid neighbors contribute with their mass.
        The intermediate density is stored in particle.density_intermediate.
        """
        for particle in self.particles:
            if particle.type_particle != "fluid":
                continue
            density_corr_fluid = 0.0
            density_corr_solid = 0.0
            for neighbor in particle.neighbors:
                grad = grad_w(particle.position, neighbor.position, self.h)
                v_diff = particle.velocity - neighbor.velocity
                dot_val = v_diff[0] * grad[0] + v_diff[1] * grad[1]
                if neighbor.type_particle == "fluid":
                    density_corr_fluid += dot_val
                elif neighbor.type_particle == "solid":
                    density_corr_solid += neighbor.mass * dot_val
            density_adv = particle.density + dt_sph * (
                particle.mass * density_corr_fluid + density_corr_solid)
            particle.density_intermediate = max(density_adv, self.rest_density)

    def adapt_velocity_density(self, dt_sph):
        """
        For each fluid particle, compute a velocity correction based on the density error.
        Compute kappa = (density_intermediate - rest_density) * alpha / (dt_sph^2)
        Then for each neighbor, accumulate a correction term:
          - Fluid neighbor: add rest_density * (kappa_i/rho_i + kappa_j/rho_j) * grad_w
          - Solid neighbor: add 2.0 * neighbor.mass * (kappa_i/rho_i) * grad_w
        Finally, update particle.velocity -= dt_sph * intermediate_velocity.
        """
        for particle in self.particles:
            if particle.type_particle != "fluid":
                continue
            rho_i = particle.density
            kappa_i = (particle.density_intermediate -
                       self.rest_density) * particle.alpha / (dt_sph * dt_sph)
            intermediate_velocity = np.zeros(2, dtype=np.float64)
            for neighbor in particle.neighbors:
                if neighbor.index == particle.index:
                    continue
                grad = grad_w(particle.position, neighbor.position, self.h)
                if neighbor.type_particle == "fluid":
                    rho_j = neighbor.density
                    kappa_j = (neighbor.density_intermediate -
                               self.rest_density) * neighbor.alpha / (dt_sph *
                                                                      dt_sph)
                    intermediate_velocity += self.rest_density * (
                        (kappa_i / rho_i) + (kappa_j / rho_j)) * grad
                elif neighbor.type_particle == "solid":
                    intermediate_velocity += 2.0 * neighbor.mass * (
                        kappa_i / rho_i) * grad
            particle.velocity -= dt_sph * intermediate_velocity

    def solve_pressure_cst_density(self,
                                   nbr_iter_max=2,
                                   threshold_density=1.0):
        """
        Iteratively solve for the pressure correction using a constant density approach.
        Uses dt_sph (set to self.dt) and iterates until the average fluid density error is below threshold.
        """
        dt_sph = self.dt
        for iter in range(nbr_iter_max):
            # Update intermediate density for fluid particles
            self.update_density_intermediate(dt_sph)
            # Compute average fluid density from intermediate values
            densities = [
                p.density_intermediate for p in self.particles
                if p.type_particle == "fluid"
            ]
            if len(densities) == 0:
                break
            avg_density = sum(densities) / len(densities)
            if abs(avg_density - self.rest_density) < threshold_density:
                break
            # Adapt velocities based on the density error
            self.adapt_velocity_density(dt_sph)

    # --- End of Constant Density Solver Methods ---

    def compute_viscosity_forces_updated(self):
        """
        Compute viscosity forces on fluid particles, taking into account both fluid and solid neighbors.
        (See earlier documentation for details.)
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
        (See earlier documentation for details.)
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
                for i in range(2):
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
            - Compute density and alpha.
            - Solve pressure using the constant density solver.
            - Compute viscosity forces.
            - (Optionally compute additional pressure forces.)
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
        # Use the constant density solver instead of the state equation:
        self.solve_pressure_cst_density(nbr_iter_max=10, threshold_density=1.0)
        # Optionally, one may compute viscosity and pressure forces further:
        self.compute_viscosity_forces_updated()
        self.compute_pressure_forces_updated()
        self.predict_intermediate_velocity()
        self.integrate()
        # self.apply_boundary_penalty()
