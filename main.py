import argparse

import numpy as np
from dfsph.drawer import SPHDrawer
from dfsph.init_helper import DFSPHInitConfig
from dfsph.particle_init import particles_init
from dfsph.particles_loader import import_snapshot
from dfsph.sim import DFSPHSim


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch a DFSPH fluid simulation."
    )

    parser.add_argument(
        "-r",
        "--support_radius",
        type=float,
        default=0.2742 / 2,
        help="SPH support radius (default: 0.2742/2)",
    )
    parser.add_argument(
        "-dt",
        "--timestep",
        type=float,
        default=0.01,
        help="Time step for the simulation (default: 0.01)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps (default: 1000)",
    )

    # Box & Grid parameters
    parser.add_argument(
        "--box_origin",
        type=float,
        nargs=2,
        default=[-2.5, -1.2],
        help="Origin of the box for particle initialization (default: -1.3"
        "-1.8)",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        nargs=2,
        default=[5, 2.5],
        help="Size of the box for particle initialization (default: 1 1)",
    )
    parser.add_argument(
        "--grid_origin",
        type=float,
        nargs=2,
        default=[-3, -1.5],
        help="Position of the grid in simulation space (default: -1.5 -2)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[6, 3.5],
        help="Grid dimensions as (width, height) (default: 3 4)",
    )

    parser.add_argument(
        "--rest_density",
        type=float,
        default=1027.0,
        help="Rest density of the fluid (default: 1027.0)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        type=str2bool,
        default=True,
        help="Enable real-time visualization (default: enabled)",
    )

    # Export and Import results for snapshots.
    parser.add_argument(
        "-e",
        "--export_results",
        type=str,
        default="",
        help="File name to export particle data (default: empty)",
    )
    parser.add_argument(
        "-i",
        "--import_results",
        type=str,
        default="",
        help="File name to import particle data for visualization (default:"
        "empty)",
    )
    parser.add_argument(
        "-ii",
        "--import_init",
        type=str,
        default="",
        help="File name to import initial particle configuration (default:"
        "empty)",
    )
    # New argument to export a GIF of the simulation.
    parser.add_argument(
        "-pg",
        "--path_gif",
        type=str,
        default="",
        help="Path to save the simulation GIF. An empty string disables GIF"
        'saving (default: "")',
    )

    args = parser.parse_args()

    # Load initial particles (either from file or by initializing)
    if args.import_init:
        print(f"Importing initial configuration from file: {args.import_init}")
        particles = import_snapshot(args.import_init, sim_time=0.0)
    else:
        particles = particles_init(
            grid_origin=args.grid_origin,
            grid_size=args.grid_size,
            h=args.support_radius,
            rest_density=args.rest_density,
            spacing=args.support_radius / 3,
            box_origin=args.box_origin,
            box_size=args.box_size,
        )

    num_particles = particles.num_particles

    if args.import_results:
        print(
            f"Loading {num_particles} particles from file:"
            f"{args.import_results}"
        )
        if args.visualize:
            drawer = SPHDrawer(
                num_particles=num_particles,
                grid_origin=args.grid_origin,
                grid_size=args.grid_size,
                support_radius=args.support_radius,
                cell_size=args.support_radius,
                import_path=args.import_results,
            )
            drawer.run(None)
    else:
        print(f"Launching DFSPH simulation with {num_particles} particles...")
        # Create DFSPH configuration using the helper class.
        config = DFSPHInitConfig(
            h=args.support_radius,
            dt=args.timestep,
            grid_origin=tuple(args.grid_origin),
            grid_size=tuple(args.grid_size),
            cell_size=args.support_radius,
            rest_density=args.rest_density,
        )
        sim = DFSPHSim(particles, config, export_path=args.export_results)

        if args.visualize:
            print("Launching a first update...")
            sim.update()
            print("Launching rendering...")
            drawer = SPHDrawer(
                num_particles=num_particles,
                grid_origin=args.grid_origin,
                grid_size=args.grid_size,
                support_radius=args.support_radius,
                cell_size=args.support_radius,
            )
            drawer.set_particles(sim.particles)

            # Check if a GIF should be created.
            gif_drawer = None
            if args.path_gif != "":
                # Import the Drawer2Gif class from your script.
                from dfsph.drawer_2_gif import Drawer2Gif

                # Duration here can be tuned; we're using the timestep as a
                # starting point.
                gif_drawer = Drawer2Gif(
                    filename=args.path_gif, duration=args.timestep
                )
                print(f"GIF recording enabled. Saving to: {args.path_gif}")

            def update_sim():
                sim.update()
                drawer.sim_time = sim.sim_time
                # Save current frame to gif if recording is enabled.
                if gif_drawer is not None:
                    gif_drawer.add_screen(drawer.screen)

            drawer.run(update_sim)

            # After quitting the simulation, close the gif writer.
            if gif_drawer is not None:
                gif_drawer.close()
        else:
            print(
                f"Starting simulation without visualization with"
                f"{num_particles} particles..."
            )
            for i in range(args.steps):
                sim.update()
                if i % 100 == 0:
                    print(f"Step {i}/{args.steps} complete.")
            print("Simulation completed.")


if __name__ == "__main__":
    main()
