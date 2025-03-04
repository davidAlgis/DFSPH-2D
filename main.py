import argparse
import numpy as np
from dfsph.sim import DFSPHSim
from dfsph.drawer import SPHDrawer
from dfsph.particle_init import particles_init
from dfsph.particles_loader import import_snapshot


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(
        description="Launch a DFSPH fluid simulation.")

    parser.add_argument("-r",
                        "--support_radius",
                        type=float,
                        default=0.2742 / 2,
                        help="SPH support radius (default: 0.2742/2)")
    parser.add_argument("-dt",
                        "--timestep",
                        type=float,
                        default=0.01,
                        help="Time step for the simulation (default: 0.01)")
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps before relaunch (default: 1000)")

    # Box & Grid parameters
    parser.add_argument(
        "--box_origin",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        help=
        "Origin of the box where particles are initialized (default: 0.5 0.5)")
    parser.add_argument(
        "--box_size",
        type=float,
        nargs=2,
        default=[2, 2],
        help="Size of the box where particles are initialized (default: 2 2)")
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[4, 4],
        help="Size of the grid as (width, height) (default: 4 4)")
    parser.add_argument(
        "--grid_origin",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Position of the grid in simulation space (default: 0.0 0.0)")

    parser.add_argument("--rest_density",
                        type=float,
                        default=1027.0,
                        help="Rest density of the fluid (default: 1027.0)")
    parser.add_argument(
        "-v",
        "--visualize",
        type=str2bool,
        default=True,
        help="Enable real-time visualization using Pygame (default: enabled)")

    # Export and Import results for snapshots.
    parser.add_argument(
        "-e",
        "--export_results",
        type=str,
        default="",
        help=
        "Relative file name to export particle data (stored as binary Parquet) (default: \"\")"
    )
    parser.add_argument(
        "-i",
        "--import_results",
        type=str,
        default="",
        help=
        "Relative file name to import particle data for visualization only (default: \"\")"
    )
    # New argument: Import initial configuration.
    parser.add_argument(
        "-ii",
        "--import_init",
        type=str,
        default="",
        help=
        "Relative file name to import initial particle configuration. If provided, simulation uses this configuration instead of generating new particles."
    )

    args = parser.parse_args()

    # If an initial configuration file is provided, load particles from it.
    if args.import_init:
        print(f"Importing initial configuration from file: {args.import_init}")
        particles = import_snapshot(args.import_init, sim_time=0.0)
    else:
        particles = particles_init(grid_origin=args.box_origin,
                                   grid_size=args.box_size,
                                   h=args.support_radius,
                                   rest_density=args.rest_density,
                                   spacing=args.support_radius / 3,
                                   box_origin=args.box_origin,
                                   box_size=args.box_size)

    num_particles = particles.num_particles

    # If import_results is provided, we're in visualization-only mode.
    if args.import_results:
        print(
            f"Loading {num_particles} particles from file: {args.import_results}"
        )
        if args.visualize:
            drawer = SPHDrawer(num_particles=num_particles,
                               grid_origin=args.grid_origin,
                               grid_size=args.grid_size,
                               support_radius=args.support_radius,
                               cell_size=args.support_radius,
                               import_path=args.import_results)
            # Run visualization; drawer will load snapshots from the file.
            drawer.run(None)
    else:
        print(f"Launching DFSPH simulation with {num_particles} particles...")
        cell_size = args.support_radius

        # Create simulation instance.
        sim = DFSPHSim(particles,
                       h=args.support_radius,
                       dt=args.timestep,
                       grid_origin=tuple(args.grid_origin),
                       grid_size=tuple(args.grid_size),
                       cell_size=cell_size,
                       rest_density=args.rest_density,
                       export_path=args.export_results)

        if args.visualize:
            drawer = SPHDrawer(num_particles=num_particles,
                               grid_origin=args.grid_origin,
                               grid_size=args.grid_size,
                               support_radius=args.support_radius,
                               cell_size=cell_size)
            drawer.set_particles(sim.particles)

            def update_sim():
                sim.update()
                drawer.sim_time = sim.sim_time

            drawer.run(update_sim)
        else:
            print(
                f"Starting simulation without visualization with {num_particles} particles..."
            )
            for i in range(args.steps):
                sim.update()
                if i % 100 == 0:
                    print(f"Step {i}/{args.steps} complete.")
            print("Simulation completed.")


if __name__ == "__main__":
    main()
