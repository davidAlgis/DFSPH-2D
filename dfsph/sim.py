import numpy as np
from dfsph.grid import Grid
from dfsph.particle import PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION
import dfsph.sph_accelerated as sphjit
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
                 water_viscosity=10,
                 surface_tension_coeff=2.0):
        self.particles = particles
        self.num_particles = len(self.particles)
        self.h = h
        self.dt = dt
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.surface_tension_coeff = surface_tension_coeff  # <---- new param

        self.mean_density = 0
        self.gamma_mass_solid = 1.4

        self.grid = Grid(grid_origin, grid_size, cell_size)
        self.gravity = np.array([0, -9.81], dtype=np.float64)

        self.preallocate_buffers()
        self.viscosity_coefficient_solid = 0.01

        self.find_neighbors()
        self.compute_density_and_alpha()
        self.update_mass_solid()

    def preallocate_buffers(self):
        N = self.num_particles
        neighbor_cap = max(1, N // 4)
        self.positions_buf = np.zeros((N, 2), dtype=np.float64)
        self.velocities_buf = np.zeros((N, 2), dtype=np.float64)
        self.masses_buf = np.zeros(N, dtype=np.float64)
        self.densities_buf = np.zeros(N, dtype=np.float64)
        self.pressures_buf = np.zeros(N, dtype=np.float64)
        self.is_solid_buf = np.zeros(N, dtype=np.int32)

        self.neighbor_counts_buf = np.zeros(N, dtype=np.int32)
        self.neighbor_starts_buf = np.zeros(N, dtype=np.int32)
        self.neighbor_indices_buf = np.zeros(neighbor_cap * N, dtype=np.int32)

    def _pack_particle_data(self):
        N = self.num_particles
        pos = self.positions_buf
        vel = self.velocities_buf
        mass = self.masses_buf
        dens = self.densities_buf
        press = self.pressures_buf
        is_solid = self.is_solid_buf
        n_list = self.neighbor_indices_buf
        n_counts = self.neighbor_counts_buf
        n_starts = self.neighbor_starts_buf

        idx_ptr = 0
        for i, p in enumerate(self.particles):
            pos[i] = p.position
            vel[i] = p.velocity
            mass[i] = p.mass
            dens[i] = p.density
            press[i] = p.pressure
            is_solid[i] = 1 if p.type_particle == 'solid' else 0

            n_counts[i] = len(p.neighbors)
            n_starts[i] = idx_ptr
            for neigh in p.neighbors:
                if idx_ptr >= n_list.shape[0]:
                    raise ValueError("Exceeded neighbor buffer!")
                n_list[idx_ptr] = neigh.index
                idx_ptr += 1
        neighbor_indices = n_list[:idx_ptr].copy()
        return pos, vel, mass, dens, press, is_solid, neighbor_indices, n_starts, n_counts

    def compute_density_and_alpha(self):
        pos, vel, mass, dens, press, is_solid, n_idx, n_starts, n_counts = self._pack_particle_data(
        )
        densities_out, alphas_out = sphjit.compute_density_alpha_numba(
            pos, mass, n_idx, n_starts, n_counts, self.h, self.rest_density)
        total = 0
        for i, p in enumerate(self.particles):
            p.density = densities_out[i]
            p.alpha = alphas_out[i]
            total += densities_out[i]
        self.mean_density = total / self.num_particles

    def compute_pressure(self, B=1000.0, gamma=7):
        for p in self.particles:
            p.pressure = B * ((p.density / self.rest_density)**gamma - 1)

    def update_mass_solid(self):
        pos, vel, mass, dens, press, is_solid, n_idx, n_starts, n_counts = self._pack_particle_data(
        )
        new_masses = sphjit.update_mass_solid_numba(pos, is_solid, n_idx,
                                                    n_starts, n_counts, self.h,
                                                    self.rest_density,
                                                    self.gamma_mass_solid,
                                                    mass.copy())
        for i, p in enumerate(self.particles):
            if is_solid[i] == 1:
                p.mass = new_masses[i]

    def compute_viscosity_forces_updated(self):
        pos, vel, mass, dens, press, is_solid, n_idx, n_starts, n_counts = self._pack_particle_data(
        )
        vis_forces = sphjit.compute_viscosity_forces_updated_numba(
            pos, vel, is_solid, dens, mass, n_idx, n_starts, n_counts, self.h,
            self.water_viscosity, self.viscosity_coefficient_solid)
        for i, p in enumerate(self.particles):
            if p.type_particle == 'fluid':
                p.add_force(VISCOSITY, vis_forces[i])

    def compute_pressure_forces_updated(self):
        pos, vel, mass, dens, press, is_solid, n_idx, n_starts, n_counts = self._pack_particle_data(
        )
        p_forces = sphjit.compute_pressure_forces_updated_numba(
            pos, is_solid, dens, press, mass, n_idx, n_starts, n_counts,
            self.h)
        for i, p in enumerate(self.particles):
            if p.type_particle == 'fluid':
                p.add_force(PRESSURE, p_forces[i])

    def compute_surface_tension_forces_updated(self):
        """
        Compute surface tension forces for fluid-fluid interactions.
        """
        pos, vel, mass, dens, press, is_solid, n_idx, n_starts, n_counts = self._pack_particle_data(
        )
        surf_forces = sphjit.compute_surface_tension_forces_updated_numba(
            pos, vel, is_solid, dens, mass, n_idx, n_starts, n_counts, self.h,
            self.surface_tension_coeff)
        for i, p in enumerate(self.particles):
            if p.type_particle == 'fluid':
                p.add_force(SURFACE_TENSION, surf_forces[i])

    def predict_intermediate_velocity(self):
        for p in self.particles:
            if p.type_particle == 'fluid':
                p.add_force(EXTERNAL, p.mass * self.gravity)
                a = p.total_force() / p.mass
                p.velocity += a * self.dt

    def integrate(self):
        for p in self.particles:
            if p.type_particle == 'fluid':
                p.position += p.velocity * self.dt

    def apply_boundary_penalty(self, collider_damping=0.5):
        bottom, top = self.get_bottom_and_top()
        if collider_damping > 0:
            collider_damping = -collider_damping
        for p in self.particles:
            if p.type_particle != 'fluid':
                continue
            for i in range(2):
                if p.position[i] < bottom[i]:
                    shift = bottom[i] - p.position[i]
                    p.position[i] = bottom[i] + min(1.0, shift)
                    p.velocity[i] *= collider_damping
                elif p.position[i] > top[i]:
                    shift = p.position[i] - top[i]
                    p.position[i] = top[i] - min(1.0, shift)
                    p.velocity[i] *= collider_damping

    def get_bottom_and_top(self):
        left = self.grid.grid_origin[0]
        low = self.grid.grid_origin[1]
        rgt = left + self.grid.grid_size[0]
        hi = low + self.grid.grid_size[1]
        return np.array([left, low],
                        dtype=np.float64), np.array([rgt, hi],
                                                    dtype=np.float64)

    def find_neighbors(self):
        self.grid.update_grid(self.particles)
        for p in self.particles:
            p.neighbors = self.grid.find_neighbors(p, self.h)

    def adapt_dt_for_cfl(self):
        vmax = 0
        for p in self.particles:
            speed = np.hypot(p.velocity[0], p.velocity[1])
            if speed > vmax:
                vmax = speed
        if vmax < 1e-6:
            self.dt = 0.033
        else:
            new_dt = 0.3999 * self.h / vmax
            self.dt = max(1e-4, min(new_dt, 0.033))

    def reset_forces(self):
        for p in self.particles:
            p.reset_forces()

    def update(self):
        """
        Full step. If you want to keep calls to pack minimal,
        do them inside each subfunction or unify them in one big call.
        """
        self.adapt_dt_for_cfl()
        self.reset_forces()
        self.find_neighbors()

        self.compute_density_and_alpha()
        self.update_mass_solid()
        self.compute_viscosity_forces_updated()
        self.compute_pressure()
        self.compute_pressure_forces_updated()
        # Add the surface tension force now
        self.compute_surface_tension_forces_updated()

        self.predict_intermediate_velocity()
        self.integrate()
        self.apply_boundary_penalty()
