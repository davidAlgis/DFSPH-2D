import argparse
import numpy as np
from dfsph.simulator import DFSPHSim
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
    parser.add_argument(
        "-n",
        "--num_particles",
        type=int,
        default=1000,
        help="Total number of particles in the simulation (default: 1000)")
    parser.add_argument("-m",
                        "--mass",
                        type=float,
                        default=1.0,
                        help="Mass of each particle (default: 1.0)")
    parser.add_argument("-r",
                        "--support_radius",
                        type=float,
                        default=0.2742,
                        help="SPH support radius (default: 1.0)")
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

    # Grid parameters
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[10, 10],
        help="Size of the grid as (width, height) (default: 50 50)")
    parser.add_argument(
        "--grid_position",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Position of the grid in simulation space (default: 0.0 0.0)")
    parser.add_argument("--cell_size",
                        type=float,
                        default=0.2742,
                        help="Size of each cell in the grid (default: 1.0)")

    # Visualization option
    parser.add_argument(
        "-v",
        "--visualize",
        type=str2bool,
        default=True,
        help="Enable real-time visualization using Vispy (default: enabled).")

    # Parse command-line arguments
    args = parser.parse_args()
    particles = particles_init(args.grid_size,
                               args.num_particles,
                               args.mass,
                               args.support_radius,
                               spacing=0.1371)

    # Create the simulation instance
    sim = DFSPHSim(particles,
                   h=args.support_radius,
                   dt=args.timestep,
                   grid_size=tuple(args.grid_size),
                   grid_position=tuple(args.grid_position),
                   cell_size=args.cell_size)

    if args.visualize:
        # Create the visualization drawer
        drawer = SPHDrawer(num_particles=len(particles),
                           grid_size=args.grid_size,
                           grid_position=args.grid_position)

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
