import os
import pygame
import numpy as np
import threading
from dfsph.drawer_ui import UIDrawer
from dfsph.kernels import w


class SPHDrawer:

    def __init__(self,
                 num_particles,
                 grid_size,
                 grid_position,
                 support_radius,
                 cell_size,
                 width=600,
                 height=600,
                 particle_radius=3,
                 density_range=(500, 1500)):
        """
        Initialize the Pygame visualization for SPH particles.
        
        :param num_particles: Maximum number of particles to display.
        :param grid_size: Tuple (grid_width, grid_height) defining the physical simulation area.
        :param grid_position: Tuple (grid_x, grid_y) defining the lower-left corner of the grid.
        :param support_radius: The SPH support radius (h).
        :param width: Width of the window.
        :param height: Height of the window.
        :param particle_radius: Radius of the particles in pixels.
        :param density_range: Tuple (min_density, max_density) for color scaling.
        """
        pygame.init()

        # Convert grid size and position to float arrays.
        self.grid_size = np.array(grid_size, dtype=float)
        self.grid_position = np.array(grid_position, dtype=float)
        self.cell_size = cell_size
        self.h = support_radius  # store the support radius

        # Set screen size based on grid size (maintaining aspect ratio).
        aspect_ratio = self.grid_size[0] / self.grid_size[1]
        self.width = width
        self.height = int(width /
                          aspect_ratio) if aspect_ratio >= 1 else height
        self.particle_radius = particle_radius

        # Compute scaling factors for world-to-screen conversion.
        self.scale_x = self.width / self.grid_size[0]
        self.scale_y = self.height / self.grid_size[1]

        # Initialize the screen.
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("DFSPH Visualization")

        # Colors.
        self.bg_color = (25, 25, 25)
        self.border_color = (255, 255, 255)  # White border.
        self.grid_color = (100, 100, 100)  # Gray grid lines.
        self.density_range = density_range

        self.particles = []  # Will be set via set_particles().
        self.running = False
        self.clock = pygame.time.Clock()

        # Create UI drawer.
        self.ui = UIDrawer(self.screen,
                           self.width,
                           self.height,
                           button_size=30,
                           padding=5)

        # Simulation control flags.
        self.paused = False
        self.step_once = False

        # For highlighting: selected particle index, its position, and neighbor indices.
        self.highlighted_index = None
        self.selected_particle_pos = None
        self.highlighted_neighbors = set()

    def set_particles(self, particles):
        """
        Update the particle positions and properties from the simulation.
        
        :param particles: List of Particle instances.
        """
        if len(particles) > 0:
            self.particles = np.array([{
                'index':
                p.index,
                'pos': (p.position[0], p.position[1]),
                'density':
                p.density,
                'alpha':
                p.alpha,
                'velocity': (p.velocity[0], p.velocity[1]),
                'neighbors': [n.index for n in p.neighbors]
            } for p in particles])

    def world_to_screen(self, world_pos):
        """
        Convert simulation world coordinates to screen space.
        
        :param world_pos: 2D tuple or array of (x, y) world position.
        :return: (x, y) screen coordinates.
        """
        screen_x = int(
            round((world_pos[0] - self.grid_position[0]) * self.scale_x))
        screen_y = int(
            round((self.grid_size[1] -
                   (world_pos[1] - self.grid_position[1])) * self.scale_y))
        return screen_x, screen_y

    def screen_to_world(self, screen_pos):
        """
        Convert screen coordinates to simulation world coordinates.
        
        :param screen_pos: (x, y) tuple from a mouse event.
        :return: 2D numpy array of (x, y) world coordinates.
        """
        x, y = screen_pos
        world_x = x / self.scale_x + self.grid_position[0]
        world_y = self.grid_position[1] + self.grid_size[1] - (y /
                                                               self.scale_y)
        return np.array([world_x, world_y])

    def draw_grid(self):
        """
        Draw the simulation grid with horizontal and vertical lines.
        The grid lines are drawn every self.cell_size world units.
        """
        # Here, assume that you want grid lines every cell_size (which can be set externally).
        # For example, you could set self.cell_size = 2 * self.h in your main script.

        num_cells_x = int(np.floor(self.grid_size[0] / self.cell_size))
        num_cells_y = int(np.floor(self.grid_size[1] / self.cell_size))

        top_left = self.world_to_screen(self.grid_position)
        bottom_right = self.world_to_screen(self.grid_position +
                                            self.grid_size)

        # Draw vertical lines.
        for i in range(num_cells_x + 1):
            world_x = self.grid_position[0] + i * self.cell_size
            screen_x, _ = self.world_to_screen(
                (world_x, self.grid_position[1]))
            pygame.draw.line(self.screen, self.grid_color,
                             (screen_x, bottom_right[1]),
                             (screen_x, top_left[1]), 1)

        # Draw horizontal lines.
        for j in range(num_cells_y + 1):
            world_y = self.grid_position[1] + j * self.cell_size
            _, screen_y = self.world_to_screen(
                (self.grid_position[0], world_y))
            pygame.draw.line(self.screen, self.grid_color,
                             (top_left[0], screen_y),
                             (bottom_right[0], screen_y), 1)

        # Draw border.
        pygame.draw.rect(self.screen, self.border_color,
                         (top_left[0], bottom_right[1], bottom_right[0] -
                          top_left[0], top_left[1] - bottom_right[1]), 2)

    def draw_buttons(self):
        """
        Draw the control buttons using the UI drawer.
        """
        self.ui.draw_buttons()

    def get_particle_color(self, density, particle_index, pos):
        """
        Interpolates a color based on the particle's density and its distance (via the kernel value)
        to the selected particle.
        
        - The selected particle is rendered in red.
        - For neighbor particles, we compute the kernel value w between the selected particle's
          position and the current particle's position. When w is high (i.e. the neighbor is very
          close), the color is red; when w is low, the color transitions to a yellow-white shade.
        - Otherwise, the color is interpolated between blue (low density) and green (high density).
        
        :param density: The density value of the particle.
        :param particle_index: Unique particle index.
        :param pos: The particle's position (world coordinates, as a tuple or list).
        :return: (R, G, B) tuple for the particle's color.
        """
        # If a particle is highlighted, adjust color based on its distance to the selected particle.
        if self.highlighted_index is not None:
            if particle_index == self.highlighted_index:
                return (255, 0, 0)  # Selected particle in red.
            elif particle_index in self.highlighted_neighbors and self.selected_particle_pos is not None:
                # Convert positions to numpy arrays.
                selected_pos = np.array(self.selected_particle_pos,
                                        dtype=float)
                current_pos = np.array(pos, dtype=float)
                w_val = w(selected_pos, current_pos, self.h)
                w_max = w(selected_pos, selected_pos, self.h)
                norm = w_val / w_max if w_max != 0 else 0
                # Interpolate between red and yellow-white:
                #   - Close neighbors (norm ~ 1) -> red (255,0,0)
                #   - Farther neighbors (norm ~ 0) -> yellow-white (255,255,200)
                low_color = np.array([255, 255, 200], dtype=float)
                high_color = np.array([255, 0, 0], dtype=float)
                color = low_color * (1 - norm) + high_color * norm
                return (int(color[0]), int(color[1]), int(color[2]))

        # Fallback: Interpolate between blue (low density) and green (high density).
        min_d, max_d = self.density_range
        normalized = np.clip((density - min_d) / (max_d - min_d), 0, 1)
        r = 0
        g = int(normalized * 255)
        b = int((1 - normalized) * 255)
        return (r, g, b)

    def draw_particles(self):
        """
        Draw all particles with colors based on density and highlighting.
        Also draws a circle of radius 2h around the selected particle.
        """
        self.screen.fill(self.bg_color)
        self.draw_grid()

        for particle in self.particles:
            screen_x, screen_y = self.world_to_screen(particle['pos'])
            color = self.get_particle_color(particle['density'],
                                            particle['index'], particle['pos'])
            pygame.draw.circle(self.screen, color, (screen_x, screen_y),
                               self.particle_radius)

        # Draw highlight circle for the selected particle.
        if self.highlighted_index is not None and self.h is not None:
            for particle in self.particles:
                if particle['index'] == self.highlighted_index:
                    center = self.world_to_screen(particle['pos'])
                    scale = min(self.scale_x, self.scale_y)
                    circle_radius = int(2 * self.h * scale)
                    pygame.draw.circle(self.screen, (255, 255, 255), center,
                                       circle_radius, 2)
                    break

        self.draw_buttons()
        pygame.display.flip()

    def handle_click(self, mouse_pos):
        """
        Handle mouse click events. If a control button is clicked, perform its action.
        Otherwise, check for particle clicks to highlight a particle and its neighbors.
        
        :param mouse_pos: (x, y) tuple from the mouse event.
        """
        ui_action = self.ui.handle_click(mouse_pos)
        if ui_action is not None:
            if ui_action == "play":
                self.paused = False
                self.step_once = False
                print("Simulation resumed (Play).")
            elif ui_action == "pause":
                self.paused = True
                self.step_once = False
                print("Simulation paused (Pause).")
            elif ui_action == "step":
                self.step_once = True
                self.paused = True
                print("Simulation stepped (Step).")
            return

        world_click = self.screen_to_world(mouse_pos)
        threshold = self.particle_radius / min(self.scale_x, self.scale_y)
        clicked_index = None
        for particle in self.particles:
            pos = np.array(particle['pos'])
            if np.linalg.norm(pos - world_click) <= threshold:
                clicked_index = particle['index']
                print("Particle clicked:")
                print(f"  Index: {particle['index']}")
                print(f"  Position: {particle['pos']}")
                print(f"  Density: {particle['density']}")
                print(f"  Alpha: {particle['alpha']}")
                print(f"  Velocity: {particle['velocity']}")
                break

        if clicked_index is not None:
            self.highlighted_index = clicked_index
            for particle in self.particles:
                if particle['index'] == clicked_index:
                    self.selected_particle_pos = particle['pos']
                    self.highlighted_neighbors = set(
                        particle.get('neighbors', []))
                    break
        else:
            self.highlighted_index = None
            self.selected_particle_pos = None
            self.highlighted_neighbors = set()

    def run(self, update_func, timestep=0.05):
        """
        Start the Pygame event loop to visualize the simulation.
        
        :param update_func: Function to update simulation at each timestep.
        :param timestep: Time interval (seconds) between each update.
        """
        self.running = True
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

            self.draw_particles()
            self.clock.tick(30)

        pygame.quit()

    def simulation_loop(self, update_func):
        """
        Runs the simulation update function in a separate thread.
        """
        while self.running:
            if not self.paused or self.step_once:
                update_func()
                if self.step_once:
                    self.step_once = False
