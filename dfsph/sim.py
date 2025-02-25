import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
import dfsph.sph_accelerated  # new calls to our Numba functions
from dfsph.kernels import w, grad_w


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
        self.particles = particles
        self.num_particles = len(self.particles)
        self.h = h
        self.dt = dt
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.mean_density = 0
        self.gamma_mass_solid = 1.4

        # Create a grid instance for neighbor search
        self.grid = Grid(grid_origin, grid_size, cell_size)
        self.gravity = np.array([0, -9.81])  # Gravity force

        self.preallocate_buffers()
        self.viscosity_coefficient_solid = 0.01

        # Initialization
        self.find_neighbors()
        self.compute_density_and_alpha()
        self.update_mass_solid()

    def preallocate_buffers(self):
        N = self.num_particles
        # Preallocate with some margin
        nbr_neighbor_max = max(1, N // 4)
        self.positions_buf = np.zeros((N, 2), dtype=np.float64)
        self.velocities_buf = np.zeros((N, 2), dtype=np.float64)
        self.masses_buf = np.zeros(N, dtype=np.float64)
        self.densities_buf = np.zeros(N, dtype=np.float64)
        self.pressures_buf = np.zeros(N, dtype=np.float64)
        self.is_solid_buf = np.zeros(N, dtype=np.int32)  # 1 if solid, 0 if fluid

        self.neighbor_counts_buf = np.zeros(N, dtype=np.int32)
        self.neighbor_starts_buf = np.zeros(N, dtype=np.int32)
        self.neighbor_indices_buf = np.zeros(nbr_neighbor_max*N, dtype=np.int32)

    def _pack_particle_data(self):
        N = self.num_particles
        pos = self.positions_buf
        vel = self.velocities_buf
        mass = self.masses_buf
        neighbor_list = self.neighbor_indices_buf
        n_counts = self.neighbor_counts_buf
        n_starts = self.neighbor_starts_buf

        # Also store densities, pressures, type (solid/fluid)
        dens = self.densities_buf
        press = self.pressures_buf
        is_solid = self.is_solid_buf

        index_ptr = 0
        for i, p in enumerate(self.particles):
            pos[i, :] = p.position
            vel[i, :] = p.velocity
            mass[i]    = p.mass
            dens[i]    = p.density
            press[i]   = p.pressure
            is_solid[i] = 1 if p.type_particle == "solid" else 0

            n_neighbors = len(p.neighbors)
            n_counts[i] = n_neighbors
            n_starts[i] = index_ptr

            for neighbor in p.neighbors:
                if index_ptr >= neighbor_list.shape[0]:
                    raise ValueError("Exceeded preallocated neighbor buffer size!")
                neighbor_list[index_ptr] = neighbor.index
                index_ptr += 1

        neighbor_indices = neighbor_list[:index_ptr].copy()
        return pos, vel, mass, dens, press, is_solid, neighbor_indices, n_starts, n_counts

    def compute_density_and_alpha(self):
        pos, _, masses, _, _, _, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data()
        densities, alphas = dfsph.sph_accelerated.compute_density_alpha_numba(
            pos, masses, neighbor_indices, neighbor_starts,
            neighbor_counts, self.h, self.rest_density)

        total_density = 0.0
        for i, p in enumerate(self.particles):
            p.density = densities[i]
            p.alpha   = alphas[i]
            total_density += densities[i]

        self.mean_density = total_density / self.num_particles

    def compute_pressure(self, B=1000.0, gamma=7):
        for p in self.particles:
            p.pressure = B*((p.density/self.rest_density)**gamma - 1)

    def update_mass_solid(self):
        """
        Moved loop to Numba function. We gather data, call Numba, then reassign.
        """
        pos, _, masses, _, _, is_solid, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data()
        updated_masses = dfsph.sph_accelerated.update_mass_solid_numba(
            pos, is_solid, neighbor_indices, neighbor_starts, neighbor_counts,
            self.h, self.rest_density, self.gamma_mass_solid, masses.copy()
        )
        # Reassign masses
        for i, p in enumerate(self.particles):
            if is_solid[i] == 1:
                p.mass = updated_masses[i]

    def compute_viscosity_forces_updated(self):
        """
        Moved core loop to Numba.
        We'll get an array of computed viscosity forces and apply them to each fluid particle.
        """
        pos, vel, masses, dens, _, is_solid, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data()

        # Call Numba function
        viscosity_forces = dfsph.sph_accelerated.compute_viscosity_forces_updated_numba(
            pos, vel, is_solid, dens, masses,
            neighbor_indices, neighbor_starts, neighbor_counts,
            self.h, self.water_viscosity, self.viscosity_coefficient_solid
        )

        # Apply the results
        for i, p in enumerate(self.particles):
            if p.type_particle == "fluid":
                p.add_force(VISCOSITY, viscosity_forces[i])

    def compute_pressure_forces_updated(self):
        """
        Moved core loop to Numba.
        We'll get an array of computed pressure forces for each fluid particle.
        """
        pos, _, masses, dens, press, is_solid, neighbor_indices, neighbor_starts, neighbor_counts = self._pack_particle_data()
        
        pressure_forces = dfsph.sph_accelerated.compute_pressure_forces_updated_numba(
            pos, is_solid, dens, press, masses,
            neighbor_indices, neighbor_starts, neighbor_counts,
            self.h
        )
        for i, p in enumerate(self.particles):
            if p.type_particle == "fluid":
                p.add_force(PRESSURE, pressure_forces[i])

    def predict_intermediate_velocity(self):
        for p in self.particles:
            if p.type_particle == "fluid":
                p.add_force(EXTERNAL, p.mass*self.gravity)
                total_force = p.total_force()
                accel = total_force / p.mass
                p.velocity += accel * self.dt

    def integrate(self):
        for p in self.particles:
            if p.type_particle == "fluid":
                p.position += p.velocity*self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        # Simple boundary
        bottom, top = self.get_bottom_and_top()
        # If you want negative damping, remove this:
        if collider_damping > 0:
            collider_damping = -collider_damping

        for p in self.particles:
            if p.type_particle != "fluid":
                continue
            for i in range(2):
                if p.position[i] < bottom[i]:
                    shift = bottom[i]-p.position[i]
                    p.position[i] = bottom[i]+min(1.0, shift)
                    p.velocity[i]*=collider_damping
                elif p.position[i]>top[i]:
                    shift = p.position[i]-top[i]
                    p.position[i] = top[i]-min(1.0, shift)
                    p.velocity[i]*=collider_damping

    def get_bottom_and_top(self):
        left  = self.grid.grid_origin[0]
        low   = self.grid.grid_origin[1]
        right = left + self.grid.grid_size[0]
        high  = low  + self.grid.grid_size[1]
        return np.array([left, low]), np.array([right, high])

    def find_neighbors(self):
        self.grid.update_grid(self.particles)
        for p in self.particles:
            p.neighbors = self.grid.find_neighbors(p, self.h)

    def adapt_dt_for_cfl(self):
        velocity_max=0
        for p in self.particles:
            speed=np.hypot(p.velocity[0], p.velocity[1])
            if speed>velocity_max:
                velocity_max=speed
        if velocity_max<1e-6:
            self.dt=0.033
        else:
            new_dt=0.3999*self.h/velocity_max
            self.dt=max(0.0001,min(new_dt,0.033))

    def update(self):
        self.adapt_dt_for_cfl()
        self.reset_forces()
        self.find_neighbors()
        self.compute_density_and_alpha()
        self.update_mass_solid()
        self.compute_viscosity_forces_updated()
        self.compute_pressure()
        self.compute_pressure_forces_updated()
        self.predict_intermediate_velocity()
        self.integrate()
        self.apply_boundary_penalty()

    def reset_forces(self):
        for p in self.particles:
            p.reset_forces()
