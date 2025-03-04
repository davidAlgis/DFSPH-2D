import os
import pygame
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import filedialog
from dfsph.drawer_ui import UIDrawer
from dfsph.kernels import w
from dfsph.particles_loader import import_snapshot, export_snapshot


class SPHDrawer:

    def __init__(self,
                 num_particles,
                 grid_origin,
                 grid_size,
                 support_radius,
                 cell_size,
                 import_path="",
                 width=600,
                 height=600,
                 particle_radius=3,
                 density_range=(500, 1500)):
        """
        Initialize the Pygame visualization for SPH particles.
        If `import_path` is set, particles will be loaded from a binary file instead of a live simulation.
        """
        pygame.init()
        self.import_path = import_path  # File path for loading snapshots
        self.grid_size = np.array(grid_size, dtype=float)
        self.grid_origin = np.array(grid_origin, dtype=float)
        self.cell_size = cell_size
        self.h = support_radius

        # Set screen size based on grid size (maintaining aspect ratio).
        aspect_ratio = self.grid_size[0] / self.grid_size[1]
        self.width = width
        self.height = int(width /
                          aspect_ratio) if aspect_ratio >= 1 else height
        self.particle_radius = particle_radius

        # Precompute scaling factors for world-to-screen conversion.
        self.scale_x = self.width / self.grid_size[0]
        self.scale_y = self.height / self.grid_size[1]

        # Initialize the screen.
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("DFSPH Visualization")

        # Colors.
        self.bg_color = (25, 25, 25)
        self.border_color = (255, 255, 255)
        self.grid_color = (100, 100, 100)
        self.density_range = density_range

        # Instead of copying particle data, store a reference.
        self.particles = None

        self.running = False
        self.clock = pygame.time.Clock()

        # Create UI drawer.
        self.ui = UIDrawer(self.screen,
                           self.width,
                           self.height,
                           button_size=30,
                           padding=5)
        # We'll use self.ui.active_button to highlight the current state.
        self.ui.active_button = None

        # Simulation control flags.
        self.paused = False
        self.step_once = False
        self.just_stepped = False

        # For highlighting.
        self.highlighted_index = None
        self.selected_particle_pos = None
        self.highlighted_neighbors = set()

        # Frame counter.
        self.frame_count = 0

        # Font for simulation time.
        self.font = pygame.font.SysFont("Arial", 18)

        # Simulation time and timestep.
        self.sim_time = 0.0
        self.timestep = 0.0333

        # Pre-render grid background.
        self.grid_surface = self._create_grid_surface()

    def _create_grid_surface(self):
        """
        Pre-render the grid lines and border to a Surface.
        """
        grid_surface = pygame.Surface((self.width, self.height))
        grid_surface.fill(self.bg_color)

        num_cells_x = int(np.floor(self.grid_size[0] / self.cell_size))
        num_cells_y = int(np.floor(self.grid_size[1] / self.cell_size))
        top_left = self.world_to_screen(self.grid_origin)
        bottom_right = self.world_to_screen(self.grid_origin + self.grid_size)

        # Draw vertical grid lines.
        for i in range(num_cells_x + 1):
            world_x = self.grid_origin[0] + i * self.cell_size
            screen_x, _ = self.world_to_screen((world_x, self.grid_origin[1]))
            pygame.draw.line(grid_surface, self.grid_color,
                             (screen_x, bottom_right[1]),
                             (screen_x, top_left[1]), 1)

        # Draw horizontal grid lines.
        for j in range(num_cells_y + 1):
            world_y = self.grid_origin[1] + j * self.cell_size
            _, screen_y = self.world_to_screen((self.grid_origin[0], world_y))
            pygame.draw.line(grid_surface, self.grid_color,
                             (top_left[0], screen_y),
                             (bottom_right[0], screen_y), 1)

        # Draw border.
        pygame.draw.rect(grid_surface, self.border_color,
                         (top_left[0], bottom_right[1], bottom_right[0] -
                          top_left[0], top_left[1] - bottom_right[1]), 2)
        return grid_surface

    def set_particles(self, particles):
        """
        Store a reference to the simulation's Particles object.
        """
        self.particles = particles

    def world_to_screen(self, world_pos):
        """
        Convert world coordinates to screen space.
        """
        if np.isnan(world_pos[0]) or np.isnan(world_pos[1]):
            return (-100, -100)
        screen_x = int(
            round((world_pos[0] - self.grid_origin[0]) * self.scale_x))
        screen_y = int(
            round((self.grid_size[1] - (world_pos[1] - self.grid_origin[1])) *
                  self.scale_y))
        return screen_x, screen_y

    def screen_to_world(self, screen_pos):
        """
        Convert screen coordinates to world coordinates.
        """
        x, y = screen_pos
        world_x = x / self.scale_x + self.grid_origin[0]
        world_y = self.grid_origin[1] + self.grid_size[1] - (y / self.scale_y)
        return np.array([world_x, world_y])

    def draw_grid(self):
        """
        Blit the pre-rendered grid background.
        """
        self.screen.blit(self.grid_surface, (0, 0))

    def draw_buttons(self):
        """
        Draw UI buttons. The UIDrawer draws a highlight border if its active_button attribute is set.
        """
        # Pass the active button to the UI drawer.
        self.ui.draw_buttons(self.ui.active_button)

    def get_particle_color(self, density, particle_index, pos):
        """
        Compute a color for a particle based on its density and highlighting.
        """
        if self.highlighted_index is not None:
            if particle_index == self.highlighted_index:
                return (255, 0, 0)
            elif particle_index in self.highlighted_neighbors and self.selected_particle_pos is not None:
                selected_pos = np.array(self.selected_particle_pos,
                                        dtype=float)
                current_pos = np.array(pos, dtype=float)
                w_val = w(selected_pos, current_pos, self.h)
                w_max = w(selected_pos, selected_pos, self.h)
                norm = w_val / w_max if w_max != 0 else 0
                low_color = np.array([255, 255, 200], dtype=float)
                high_color = np.array([255, 0, 0], dtype=float)
                color = low_color * (1 - norm) + high_color * norm
                return (int(color[0]), int(color[1]), int(color[2]))
        min_d, max_d = self.density_range
        normalized = np.clip((density - min_d) / (max_d - min_d), 0, 1)
        r = 0
        g = int(normalized * 255)
        b = int((1 - normalized) * 255)
        return (r, g, b)

    def draw_particles(self):
        """
        Render particles using the Particles object's arrays.
        """
        self.screen.blit(self.grid_surface, (0, 0))
        if self.particles is None:
            return

        n = self.particles.num_particles
        for i in range(n):
            pos = self.particles.position[i]
            screen_x, screen_y = self.world_to_screen(pos)
            if self.particles.types[i] != 0:
                color = (150, 75, 0)
            else:
                density = self.particles.density[i]
                color = self.get_particle_color(density, i, pos)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y),
                               self.particle_radius)

        # Draw a highlight circle for the selected particle.
        if self.highlighted_index is not None and self.h is not None:
            pos = self.particles.position[self.highlighted_index]
            center = self.world_to_screen(pos)
            scale = min(self.scale_x, self.scale_y)
            circle_radius = int(self.h * scale)
            pygame.draw.circle(self.screen, (255, 255, 255), center,
                               circle_radius, 2)

        self.draw_buttons()

        # Render simulation time at the top left corner.
        time_text = self.font.render(f"Time: {self.sim_time:.3f} s", True,
                                     (255, 255, 255))
        self.screen.blit(time_text, (10, 10))

        pygame.display.flip()

    def print_highlighted_particle_info(self):
        """
        Print info of the highlighted particle.
        """
        if self.particles is None or self.highlighted_index is None:
            return

        i = self.highlighted_index
        pos = self.particles.position[i]
        density = self.particles.density[i]
        mass = self.particles.mass[i]
        alpha = self.particles.alpha[i]
        vel = self.particles.velocity[i]
        cnt = self.particles.neighbor_counts[i] if hasattr(
            self.particles, 'neighbor_counts') else 0

        vf = self.particles.viscosity_forces[i]
        ef = self.particles.external_forces[i]
        pf = self.particles.pressure_forces[i]
        sf = self.particles.surface_tension_forces[i]
        total_force = vf + ef + pf + sf

        print(f"\nFrame {self.frame_count}:")
        print(f"  Index: {i}")
        print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f})")
        print(f"  Density: {density:.3f}")
        print(f"  Mass: {mass:.3f}")
        print(f"  Alpha: {alpha:.3f}")
        print(f"  Velocity: ({vel[0]:.3f}, {vel[1]:.3f})")
        print(f"  Neighbors: {cnt}")
        print(f"  Viscosity Force: ({vf[0]:.3f}, {vf[1]:.3f})")
        print(f"  External Force: ({ef[0]:.3f}, {ef[1]:.3f})")
        print(f"  Pressure Force: ({pf[0]:.3f}, {pf[1]:.3f})")
        print(f"  Surface Tension Force: ({sf[0]:.3f}, {sf[1]:.3f})")
        print(f"  Total Force: ({total_force[0]:.3f}, {total_force[1]:.3f})")

    def handle_click(self, mouse_pos):
        """
        Process mouse clicks for UI or particle selection.
        """
        ui_action = self.ui.handle_click(mouse_pos)
        if ui_action is not None:
            if ui_action == "play":
                self.paused = False
                self.step_once = False
                self.ui.active_button = "play"
                print("Simulation resumed (Play).")
            elif ui_action == "pause":
                self.paused = True
                self.step_once = False
                self.ui.active_button = "pause"
                print("Simulation paused (Pause).")
            elif ui_action == "step":
                self.step_once = True
                self.paused = True
                self.ui.active_button = "step"
                print("Simulation stepped (Step).")
            elif ui_action == "save":
                self.paused = True
                self.ui.active_button = "save"
                print("Save button clicked.")
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.asksaveasfilename(
                    title="Save Snapshot",
                    defaultextension=".parquet",
                    filetypes=[("Parquet files", "*.parquet"),
                               ("All files", "*.*")])
                root.destroy()
                if file_path:
                    export_snapshot(self.particles, file_path, self.sim_time)
                    print(f"Snapshot saved to {file_path}")
                else:
                    print("Save cancelled.")
            return

        # Particle selection logic.
        world_click = self.screen_to_world(mouse_pos)
        threshold = self.particle_radius / min(self.scale_x, self.scale_y)
        if self.particles is None:
            return

        clicked_index = None
        for i in range(self.particles.num_particles):
            pos = self.particles.position[i]
            if np.linalg.norm(np.array(pos) - world_click) <= threshold:
                clicked_index = i
                break

        if clicked_index is None:
            self.highlighted_index = None
            self.selected_particle_pos = None
            self.highlighted_neighbors = set()
            return

        self.highlighted_index = clicked_index
        self.selected_particle_pos = self.particles.position[clicked_index]
        if (hasattr(self.particles, 'neighbor_indices')
                and hasattr(self.particles, 'neighbor_starts')
                and hasattr(self.particles, 'neighbor_counts')):
            start = self.particles.neighbor_starts[clicked_index]
            cnt = self.particles.neighbor_counts[clicked_index]
            neighbors = self.particles.neighbor_indices[start:start + cnt]
            self.highlighted_neighbors = set(neighbors.tolist())
        else:
            self.highlighted_neighbors = set()

        print("\nParticle clicked:")
        print(f"  Index: {clicked_index}")
        pos = self.particles.position[clicked_index]
        print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f})")
        print(f"  Density: {self.particles.density[clicked_index]:.3f}")
        print(f"  Mass: {self.particles.mass[clicked_index]:.3f}")
        print(f"  Alpha: {self.particles.alpha[clicked_index]:.3f}")
        vel = self.particles.velocity[clicked_index]
        print(f"  Velocity: ({vel[0]:.3f}, {vel[1]:.3f})")
        cnt = self.particles.neighbor_counts[clicked_index] if hasattr(
            self.particles, 'neighbor_counts') else 0
        print(f"  Neighbors: {cnt}")
        # Also print forces immediately upon click.
        vf = self.particles.viscosity_forces[clicked_index]
        ef = self.particles.external_forces[clicked_index]
        pf = self.particles.pressure_forces[clicked_index]
        sf = self.particles.surface_tension_forces[clicked_index]
        total_force = vf + ef + pf + sf
        print(f"  Viscosity Force: ({vf[0]:.3f}, {vf[1]:.3f})")
        print(f"  External Force: ({ef[0]:.3f}, {ef[1]:.3f})")
        print(f"  Pressure Force: ({pf[0]:.3f}, {pf[1]:.3f})")
        print(f"  Surface Tension Force: ({sf[0]:.3f}, {sf[1]:.3f})")
        print(f"  Total Force: ({total_force[0]:.3f}, {total_force[1]:.3f})")

    def run(self, update_func, timestep=0.033):
        """
        Start the Pygame event loop and run the simulation update in a separate thread.
        """
        self.running = True
        self.timestep = timestep
        if self.import_path:
            self.imported_simulation_loop()
            return
        # Launch the simulation update thread.
        sim_thread = threading.Thread(target=self.simulation_loop,
                                      args=(update_func, ),
                                      daemon=True)
        sim_thread.start()

        while self.running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            # Update active state based on simulation flags.
            self.ui.active_button = "play" if not self.paused else "pause"
            self.draw_particles()

            if not self.paused:
                self.print_highlighted_particle_info()
                self.frame_count += 1
            elif self.just_stepped:
                self.print_highlighted_particle_info()
                self.frame_count += 1
                self.just_stepped = False
            self.clock.tick(30)

        pygame.quit()

    def imported_simulation_loop(self):
        """
        Loop that loads snapshots from the binary file at each timestep.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            particles = import_snapshot(self.import_path, self.sim_time)
            self.set_particles(particles)
            self.draw_particles()
            self.sim_time += self.timestep
            if not self.paused:
                self.print_highlighted_particle_info()
                self.frame_count += 1
            elif self.just_stepped:
                self.print_highlighted_particle_info()
                self.frame_count += 1
                self.just_stepped = False
            self.clock.tick(30)
        pygame.quit()

    def simulation_loop(self, update_func):
        """
        Run the simulation update function in a separate thread.
        """
        while self.running:
            if not self.paused or self.step_once:
                update_func()
                if self.step_once:
                    self.step_once = False
                    self.just_stepped = True
