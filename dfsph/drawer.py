import os
import pygame
import numpy as np
import threading
from dfsph.drawer_ui import UIDrawer


class SPHDrawer:

    def __init__(self,
                 num_particles,
                 grid_size,
                 grid_position,
                 width=600,
                 height=600,
                 particle_radius=3,
                 density_range=(500, 1500)):
        """
        Initialize the Pygame visualization for SPH particles.
        
        :param num_particles: Maximum number of particles to display.
        :param grid_size: Tuple (grid_width, grid_height) defining the physical simulation area.
        :param grid_position: Tuple (grid_x, grid_y) defining the lower-left corner of the grid.
        :param width: Width of the window.
        :param height: Height of the window.
        :param particle_radius: Radius of the particles in pixels.
        :param density_range: Tuple (min_density, max_density) for color scaling.
        """
        pygame.init()

        # Convert grid size and position to float arrays.
        self.grid_size = np.array(grid_size, dtype=float)
        self.grid_position = np.array(grid_position, dtype=float)

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

        # For highlighting clicked particle.
        self.highlighted_index = None
        self.highlighted_neighbors = set()

    def set_particles(self, particles):
        """
        Update the particle positions and properties from the simulation.
        
        :param particles: List of Particle instances.
        """
        if len(particles) > 0:
            # Store additional neighbor indices.
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
        screen_x = int((world_pos[0] - self.grid_position[0]) * self.scale_x)
        screen_y = int((self.grid_size[1] -
                        (world_pos[1] - self.grid_position[1])) * self.scale_y)
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
        """
        top_left = self.world_to_screen(self.grid_position)
        bottom_right = self.world_to_screen(self.grid_position +
                                            self.grid_size)

        for i in range(int(self.grid_size[0]) + 1):
            x = top_left[0] + i * self.scale_x
            pygame.draw.line(self.screen, self.grid_color,
                             (x, bottom_right[1]), (x, top_left[1]), 1)

        for j in range(int(self.grid_size[1]) + 1):
            y = bottom_right[1] + j * self.scale_y
            pygame.draw.line(self.screen, self.grid_color, (top_left[0], y),
                             (bottom_right[0], y), 1)

        pygame.draw.rect(self.screen, self.border_color,
                         (top_left[0], bottom_right[1], bottom_right[0] -
                          top_left[0], top_left[1] - bottom_right[1]), 2)

    def get_particle_color(self, density, particle_index):
        """
        Interpolates a color based on the particle's density. Also applies highlighting:
          - Highlighted particle: Red.
          - Highlighted neighbor: Yellow.
        Otherwise, interpolate between blue (low density) and green (high density).
        
        :param density: The density value of the particle.
        :param particle_index: The unique index of the particle.
        :return: (R, G, B) tuple for color.
        """
        if self.highlighted_index is not None:
            if particle_index == self.highlighted_index:
                return (255, 0, 0)  # Red for the clicked particle.
            elif particle_index in self.highlighted_neighbors:
                return (255, 255, 0)  # Yellow for its neighbors.
        min_d, max_d = self.density_range
        normalized = np.clip((density - min_d) / (max_d - min_d), 0, 1)
        r = 0
        g = int(normalized * 255)
        b = int((1 - normalized) * 255)
        return (r, g, b)

    def draw_particles(self):
        """
        Efficiently draws all particles onto the screen with colors based on density.
        """
        self.screen.fill(self.bg_color)
        self.draw_grid()

        for particle in self.particles:
            screen_x, screen_y = self.world_to_screen(particle['pos'])
            color = self.get_particle_color(particle['density'],
                                            particle['index'])
            pygame.draw.circle(self.screen, color, (screen_x, screen_y),
                               self.particle_radius)

        self.ui.draw_buttons()
        pygame.display.flip()

    def handle_click(self, mouse_pos):
        """
        Handle mouse click events. First check if a control button was clicked;
        otherwise, check for particle clicks.
        
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

        # If click is not on a button, check for particle selection.
        world_click = self.screen_to_world(mouse_pos)
        threshold = self.particle_radius / min(self.scale_x, self.scale_y)
        clicked_index = None
        for particle in self.particles:
            pos = np.array(particle['pos'])
            distance = np.linalg.norm(pos - world_click)
            if distance <= threshold:
                clicked_index = particle['index']
                print("Particle clicked:")
                print(f"  Index: {particle['index']}")
                print(f"  Position: {particle['pos']}")
                print(f"  Density: {particle['density']}")
                print(f"  Alpha: {particle['alpha']}")
                print(f"  Velocity: {particle['velocity']}")
                break

        # Update highlighted particle and its neighbors.
        if clicked_index is not None:
            self.highlighted_index = clicked_index
            # Find the corresponding particle in our array and set its neighbors.
            for particle in self.particles:
                if particle['index'] == clicked_index:
                    self.highlighted_neighbors = set(
                        particle.get('neighbors', []))
                    break
        else:
            # Clear highlighting if click not on any particle.
            self.highlighted_index = None
            self.highlighted_neighbors = set()

    def run(self, update_func, timestep=0.05):
        """
        Starts the Pygame event loop to visualize the simulation.
        
        :param update_func: Function to update the simulation at each timestep.
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
