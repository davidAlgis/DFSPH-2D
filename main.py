import argparse
import numpy as np
from dfsph.sim import DFSPHSim
from dfsph.drawer import SPHDrawer
from dfsph.particle_init import particles_init


def str2bool(v):
    """
    Converts a string argument to a boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(
        description="Launch a DFSPH fluid simulation.")

    # Simulation parameters
    parser.add_argument("-r",
                        "--support_radius",
                        type=float,
                        default=0.2742,
                        help="SPH support radius (default: 0.2742)")
    parser.add_argument("-dt",
                        "--timestep",
                        type=float,
                        default=0.01,
                        help="Time step for the simulation (default: 0.01)")
    parser.add_argument("-s",
                        "--steps",
                        type=int,
                        default=1000,
                        help="Number of simulation steps (default: 1000)")

    # Box parameters for particle initialization
    parser.add_argument(
        "--box_origin",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        help=
        "Origin of the box where particles are initialized (default: 0.0 0.0)")
    parser.add_argument(
        "--box_size",
        type=float,
        nargs=2,
        default=[2, 2],
        help=
        "Size of the box where particles are initialized (default: 10.0 10.0)")
    parser.add_argument("--rest_density",
                        type=float,
                        default=1027.0,
                        help="Rest density of the fluid (default: 1027 kg/m³)")

    # Grid parameters
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[4, 4],
        help="Size of the grid as (width, height) (default: 50 50)")
    parser.add_argument(
        "--grid_origin",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Position of the grid in simulation space (default: 0.0 0.0)")

    # Visualization option
    parser.add_argument(
        "-v",
        "--visualize",
        type=str2bool,
        default=True,
        help="Enable real-time visualization using Pygame (default: enabled).")

    # Parse command-line arguments
    args = parser.parse_args()

    # Initialize particles
    particles = particles_init(grid_origin=args.grid_origin,
                               grid_size=args.grid_size,
                               h=args.support_radius,
                               rest_density=args.rest_density,
                               spacing=args.support_radius / 2.5,
                               box_origin=args.box_origin,
                               box_size=args.box_size)
    num_particles = len(particles)
    print(f"Launch simulation DFSPH with {num_particles} particles...")
    cell_size = args.support_radius
    # Create the simulation instance
    sim = DFSPHSim(particles,
                   h=args.support_radius,
                   dt=args.timestep,
                   grid_origin=tuple(args.grid_origin),
                   grid_size=tuple(args.grid_size),
                   cell_size=cell_size)

    if args.visualize:
        # Create the visualization drawer
        drawer = SPHDrawer(num_particles=num_particles,
                           grid_origin=args.grid_origin,
                           grid_size=args.grid_size,
                           support_radius=args.support_radius,
                           cell_size=cell_size)

        # Define an update function for simulation
        def update_sim():
            sim.update()
            drawer.set_particles(sim.particles)

        # Run the visualization loop
        drawer.run(update_sim)

    else:
        print(f"Starting simulation with {len(particles)} particles...")
        for i in range(args.steps):
            sim.update()
            if i % 100 == 0:
                print(f"Step {i}/{args.steps} complete.")
        print("Simulation completed.")


if __name__ == "__main__":
    main()
