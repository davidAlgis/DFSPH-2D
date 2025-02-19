import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho


class SPHDrawer(app.Canvas):
    def __init__(self, sim, particle_radius=0.02):
        """
        Initialize the Vispy visualization for SPH particles.

        :param sim: Instance of DFSPHSim (the simulation object).
        :param particle_radius: Radius of the particles in the visualization.
        """
        app.Canvas.__init__(self, title='DFSPH Visualization', keys='interactive', size=(800, 800))

        self.sim = sim  # The DFSPH simulation instance
        self.particle_radius = particle_radius

        # OpenGL shader programs
        self.program = gloo.Program(vertex_shader, fragment_shader)

        # Particle positions (initialized to zeros)
        self.data = np.zeros(len(self.sim.particles), dtype=[('a_position', np.float32, 2)])

        # Initialize GPU buffer
        self.program.bind(gloo.VertexBuffer(self.data))

        # Set up orthographic projection matrix
        self.view_matrix = ortho(0, 10, 0, 10, -1, 1)
        self.program['u_matrix'] = self.view_matrix

        # OpenGL settings
        gloo.set_clear_color((0.1, 0.1, 0.1, 1.0))  # Background color
        gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def update_particles(self):
        """
        Update particle positions from the simulation.
        """
        for i, particle in enumerate(self.sim.particles):
            self.data[i]['a_position'] = particle.position

        # Update GPU buffer
        self.program.bind(gloo.VertexBuffer(self.data))

        # Redraw canvas
        self.update()

    def on_draw(self, event):
        """
        Render the scene.
        """
        gloo.clear()
        self.program.draw('points')

    def run_simulation(self, num_steps=1000):
        """
        Run the SPH simulation with visualization.

        :param num_steps: Number of simulation steps.
        """
        for step in range(num_steps):
            self.sim.update()
            self.update_particles()
            app.process_events()


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

if __name__ == '__main__':
    from dfsph import DFSPHSim

    # Initialize the simulation
    sim = DFSPHSim(
        num_particles=500,
        h=1.0,
        mass=1.0,
        dt=0.01,
        grid_size=(50, 50),
        grid_position=(0, 0),
        cell_size=1.0
    )

    # Create visualization and run simulation
    drawer = SPHDrawer(sim)
    drawer.run_simulation(1000)
