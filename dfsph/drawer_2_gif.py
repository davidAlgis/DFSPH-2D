import imageio
import numpy as np
import pygame


class Drawer2Gif:
    """
    A class to create a GIF by capturing frames from a Pygame screen.
    Each time add_screen is called, the content of the passed Pygame
    surface is appended as a frame to the GIF.
    """

    def __init__(self, filename="output.gif", duration=0.1):
        """
        Initialize the GIF writer.

        Args:
            filename (str): The output filename of the GIF.
            duration (float): Duration per frame in seconds.
        """
        self.filename = filename
        self.duration = duration
        # Create a writer that will compile frames into a GIF.
        self.writer = imageio.get_writer(
            self.filename, mode="I", duration=self.duration
        )
        print(f"GIF writer created for file: {self.filename}")

    def add_screen(self, screen):
        # Only capture the frame if the display is still initialized.
        if not pygame.display.get_init():
            return
        frame_array = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame_array, (1, 0, 2))
        self.writer.append_data(frame)

    def close(self):
        """
        Finalize the GIF and close the writer.
        """
        self.writer.close()
        print(f"GIF file '{self.filename}' closed and saved.")