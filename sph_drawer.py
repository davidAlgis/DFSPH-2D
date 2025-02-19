import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho

# Vertex Shader
vertex_shader = """
#version 120
attribute vec2 a_position;
uniform mat4 u_matrix;
void main() {
    gl_Position = u_matrix * vec4(a_position, 0.0, 1.0);
    gl_PointSize = 5.0;
}
"""

# Fragment Shader
fragment_shader = """
#version 120
void main() {
    gl_FragColor = vec4(0.2, 0.6, 1.0, 1.0);  // Blue particles
}
"""


class SPHDrawer(app.Canvas):

    def __init__(self, num_particles, particle_radius=0.02):
        """
        Initialize the Vispy visualization for SPH particles.
        
        :param num_particles: The maximum number of particles to display.
        :param particle_radius: Radius of the particles in the visualization.
        """
        app.Canvas.__init__(self,
                            title='DFSPH Visualization',
                            keys='interactive',
                            size=(600, 600))
        self.num_particles = num_particles
        self.particle_radius = particle_radius

        # Create the OpenGL program with shaders
        self.program = gloo.Program(vertex_shader, fragment_shader)

        # Allocate a structured array for particle positions
        self.data = np.zeros(self.num_particles,
                             dtype=[('a_position', np.float32, 2)])

        # Create and bind the GPU buffer
        self.vertex_buffer = gloo.VertexBuffer(self.data)
        self.program.bind(self.vertex_buffer)

        # Set up an orthographic projection matrix for the view
        self.view_matrix = ortho(0, 10, 0, 10, -1, 1)
        self.program['u_matrix'] = self.view_matrix

        # OpenGL settings: clear color and blending
        gloo.set_clear_color((0.1, 0.1, 0.1, 1.0))
        gloo.set_state(blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._timer = None
        self.show()

    def set_particles(self, particles):
        """
        Update the visualization with a new list of particles.
        
        :param particles: List of Particle instances.
        """
        # Update our data array with particle positions.
        # Only update as many particles as available (up to num_particles).
        for i, particle in enumerate(particles):
            if i < self.num_particles:
                self.data[i]['a_position'] = particle.position
        # Update the GPU buffer with the new data.
        self.vertex_buffer.set_data(self.data)
        # Request a redraw of the canvas.
        self.update()

    def on_draw(self, event):
        """
        Render the scene.
        """
        gloo.clear()
        self.program.draw('points')

    def add_update(self, timestep, update):
        # Create a Vispy timer to call the update function every timestep
        timer = app.Timer(interval=timestep, connect=update, start=True)
        # Store the timer as an attribute to prevent it from being garbage collected.
        self._timer = timer

    def launch_update(self):

        print("Starting simulation with visualization...")
        app.run()  # Launch the Vispy event loop


if __name__ == '__main__':
    # This block is for testing the drawer independently.
    # It creates dummy particles and updates the drawer periodically.
    from time import sleep

    # Dummy Particle class for testing (replace with your actual Particle class)
    class Particle:

        def __init__(self, position):
            self.position = np.array(position, dtype=float)

    # Create a list of 10 dummy particles with random positions.
    num_particles = 10
    particles = [
        Particle(np.random.rand(2) * 10) for _ in range(num_particles)
    ]

    # Initialize the SPHDrawer with the number of particles.
    drawer = SPHDrawer(num_particles=num_particles)

    # Simple function to update particle positions randomly.
    def update_test(event):
        global particles
        # Randomly perturb each particle's position.
        for p in particles:
            p.position += np.random.uniform(-0.1, 0.1, 2)
        drawer.set_particles(particles)

    # Set up a timer to trigger updates.
    timer = app.Timer(interval=0.05, connect=update_test, start=True)

    app.run()
