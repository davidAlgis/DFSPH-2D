import os
import pygame
import numpy as np


class UIDrawer:

    def __init__(self,
                 screen,
                 window_width,
                 window_height,
                 button_size=30,
                 padding=5):
        """
        Initialize the UI drawer responsible for drawing control buttons.
        
        :param screen: Pygame display surface.
        :param window_width: Width of the window.
        :param window_height: Height of the window.
        :param button_size: Size of each square button (in pixels).
        :param padding: Padding between buttons.
        """
        self.screen = screen
        self.window_width = window_width
        self.window_height = window_height
        self.button_size = button_size
        self.padding = padding

        # Define assets directory relative to this file.
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
        # Load button background and icon images.
        self.button_bg = pygame.image.load(
            os.path.join(assets_dir,
                         "button_square_depth_gradient.png")).convert_alpha()
        self.play_icon = pygame.image.load(os.path.join(
            assets_dir, "play.png")).convert_alpha()
        self.pause_icon = pygame.image.load(
            os.path.join(assets_dir, "pause.png")).convert_alpha()
        self.step_icon = pygame.image.load(os.path.join(
            assets_dir, "step.png")).convert_alpha()
        self.save_icon = pygame.image.load(os.path.join(
            assets_dir, "save.png")).convert_alpha()

        # Scale the button background to fill the square.
        self.button_bg = pygame.transform.scale(self.button_bg,
                                                (button_size, button_size))
        # Scale icons to 80% of the button size.
        icon_size = int(button_size * 0.8)
        self.play_icon = pygame.transform.scale(self.play_icon,
                                                (icon_size, icon_size))
        self.pause_icon = pygame.transform.scale(self.pause_icon,
                                                 (icon_size, icon_size))
        self.step_icon = pygame.transform.scale(self.step_icon,
                                                (icon_size, icon_size))
        self.save_icon = pygame.transform.scale(self.save_icon,
                                                (icon_size, icon_size))
        # Calculate offset to center the icon.
        self.icon_offset = ((button_size - icon_size) // 2,
                            (button_size - icon_size) // 2)

        # Define button rectangles arranged horizontally in the top-right corner.
        total_width = 4 * button_size + 5 * padding  # Now 4 buttons
        x0 = window_width - total_width
        y0 = padding
        self.play_button = pygame.Rect(x0, y0, button_size, button_size)
        self.pause_button = pygame.Rect(x0 + button_size + padding, y0,
                                        button_size, button_size)
        self.step_button = pygame.Rect(x0 + 2 * (button_size + padding), y0,
                                       button_size, button_size)
        self.save_button = pygame.Rect(x0 + 3 * (button_size + padding), y0,
                                       button_size, button_size)

    def draw_buttons(self):
        """
        Draws the control buttons with the button background and centered icons.
        """
        # Draw background and icon for Play button.
        self.screen.blit(self.button_bg, self.play_button.topleft)
        play_pos = (self.play_button.left + self.icon_offset[0],
                    self.play_button.top + self.icon_offset[1])
        self.screen.blit(self.play_icon, play_pos)

        # Draw background and icon for Pause button.
        self.screen.blit(self.button_bg, self.pause_button.topleft)
        pause_pos = (self.pause_button.left + self.icon_offset[0],
                     self.pause_button.top + self.icon_offset[1])
        self.screen.blit(self.pause_icon, pause_pos)

        # Draw background and icon for Step button.
        self.screen.blit(self.button_bg, self.step_button.topleft)
        step_pos = (self.step_button.left + self.icon_offset[0],
                    self.step_button.top + self.icon_offset[1])
        self.screen.blit(self.step_icon, step_pos)

        # Draw background and icon for Save button.
        self.screen.blit(self.button_bg, self.save_button.topleft)
        save_pos = (self.save_button.left + self.icon_offset[0],
                    self.save_button.top + self.icon_offset[1])
        self.screen.blit(self.save_icon, save_pos)

    def handle_click(self, mouse_pos):
        """
        Checks if the mouse click is on one of the control buttons.
        
        :param mouse_pos: (x, y) tuple from the mouse event.
        :return: "play", "pause", "step", "save", or None if no button was clicked.
        """
        if self.play_button.collidepoint(mouse_pos):
            return "play"
        elif self.pause_button.collidepoint(mouse_pos):
            return "pause"
        elif self.step_button.collidepoint(mouse_pos):
            return "step"
        elif self.save_button.collidepoint(mouse_pos):
            return "save"
        return None
