import pygame
import numpy as np
import threading


class SPHDrawer:

    def __init__(self,
                 num_particles,
                 width=600,
                 height=600,
                 particle_radius=3):
        pygame.init()
        self.width = width
        self.height = height
        self.particle_radius = particle_radius
        self.num_particles = num_particles

        # Initialize the screen with hardware acceleration
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("DFSPH Visualization")

        # Background and particle colors
        self.bg_color = (25, 25, 25)
        self.particle_color = (50, 150, 255)

        # Pre-rendered particle for faster drawing
        self.particle_surf = pygame.Surface(
            (self.particle_radius * 2, self.particle_radius * 2),
            pygame.SRCALPHA)
        pygame.draw.circle(self.particle_surf, self.particle_color,
                           (self.particle_radius, self.particle_radius),
                           self.particle_radius)

        self.particles = []
        self.running = False
        self.clock = pygame.time.Clock()

    def set_particles(self, particles):
        if len(particles) > 0:
            self.particles = np.array([(p.position[0], p.position[1])
                                       for p in particles])

    def draw_particles(self):
        self.screen.fill(self.bg_color)  # Clear screen

        for x, y in self.particles:
            screen_x = int((x / 10) * self.width)
            screen_y = int((1 - y / 10) * self.height)
            self.screen.blit(self.particle_surf,
                             (screen_x - self.particle_radius,
                              screen_y - self.particle_radius))

        pygame.display.flip()

    def run(self, update_func, timestep=0.05):
        self.running = True

        # Run simulation in a separate thread
        sim_thread = threading.Thread(target=self.simulation_loop,
                                      args=(update_func, ),
                                      daemon=True)
        sim_thread.start()

        while self.running:
            for event in pygame.event.get(pygame.QUIT):
                self.running = False

            self.draw_particles()
            self.clock.tick(60)  # Limit FPS to 30

        pygame.quit()

    def simulation_loop(self, update_func):
        """
        Runs the simulation in a separate thread.
        """
        while self.running:
            update_func()
