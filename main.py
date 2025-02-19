import argparse
import numpy as np
from dfsph import DFSPHSim  # Import the DFSPHSim class
from grid import Grid  # Ensure the grid system is available
from particle import Particle  # Ensure particles are correctly loaded


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
        help="Number of particles in the simulation (default: 1000)")
    parser.add_argument("-m",
                        "--mass",
                        type=float,
                        default=1.0,
                        help="Mass of each particle (default: 1.0)")
    parser.add_argument(
        "-r",
        "--support_radius",
        type=float,
        default=1.0,
        help="SPH support radius (default: 1.0)")  # Changed from -h to -r
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
        default=[50, 50],
        help="Size of the grid as (width, height) (default: 50 50)")
    parser.add_argument(
        "--grid_position",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Position of the grid in simulation space (default: 0.0 0.0)")
    parser.add_argument("--cell_size",
                        type=float,
                        default=1.0,
                        help="Size of each cell in the grid (default: 1.0)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Create and launch the DFSPH simulation
    sim = DFSPHSim(
        num_particles=args.num_particles,
        h=args.support_radius,  # Updated variable name
        mass=args.mass,
        dt=args.timestep,
        grid_size=tuple(args.grid_size),
        grid_position=tuple(args.grid_position),
        cell_size=args.cell_size)

    print(f"Starting simulation with {args.num_particles} particles...")

    # Run the simulation
    sim.run(args.steps)

    print("Simulation completed.")


if __name__ == "__main__":
    main()
