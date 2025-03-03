import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def init_export(export_path):
    """
    Initialize the export CSV file with a header.
    """
    header = "sim_time,particle_index,x,y,vx,vy,density,mass,alpha,type\n"
    with open(export_path, "w", newline="") as f:
        f.write(header)


def export_snapshot(particles, export_path, sim_time, num_workers=4):
    """
    Export a snapshot of the simulation's particle data to a CSV file.
    The snapshot is exported in parallel by partitioning the data rows.
    
    Each row contains:
        sim_time, particle_index, x, y, vx, vy, density, mass, alpha, type
    """
    num_particles = particles.num_particles
    # Prepare the data columns.
    indices = np.arange(num_particles, dtype=np.int32).reshape(-1, 1)
    sim_time_col = np.full((num_particles, 1), sim_time, dtype=np.float64)
    positions = particles.position.astype(np.float64)  # shape (n,2)
    velocities = particles.velocity.astype(np.float64)  # shape (n,2)
    density = particles.density.astype(np.float64).reshape(-1, 1)
    mass = particles.mass.astype(np.float64).reshape(-1, 1)
    alpha = particles.alpha.astype(np.float64).reshape(-1, 1)
    types = particles.types.reshape(-1, 1)

    # Stack columns horizontally: result shape (num_particles, 10)
    data = np.hstack((sim_time_col, indices, positions, velocities, density,
                      mass, alpha, types))

    # Function to convert a chunk of rows into CSV-formatted lines.
    def chunk_to_lines(chunk):
        lines = []
        fmt = "{:.5f},{:d},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:d}"
        for row in chunk:
            line = fmt.format(row[0], int(row[1]), row[2], row[3], row[4],
                              row[5], row[6], row[7], row[8], int(row[9]))
            lines.append(line)
        return lines

    # Partition the data into chunks for parallel processing.
    chunk_size = (num_particles + num_workers - 1) // num_workers
    chunks = [
        data[i:i + chunk_size] for i in range(0, num_particles, chunk_size)
    ]

    # Process each chunk in parallel.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(chunk_to_lines, chunk) for chunk in chunks]
        lines_all = []
        for future in futures:
            lines_all.extend(future.result())

    # Append all the formatted CSV lines to the export file.
    with open(export_path, "a", newline="") as f:
        for line in lines_all:
            f.write(line + "\n")


def import_snapshot(import_path, sim_time, num_workers=16):
    """
    Import a snapshot of the particles at the closest available `sim_time` from a CSV file.

    Parameters:
    - import_path (str): Path to the CSV file.
    - sim_time (float): The target simulation time. The function will pick the closest available time.
    - num_workers (int): Number of parallel workers for faster reading.

    Returns:
    - particles (Particles): A `Particles` object reconstructed at the closest `sim_time`.
    """

    print(
        f"[Import] Loading snapshot from '{import_path}' for sim_time ≈ {sim_time:.3f}..."
    )

    # Read CSV file
    df = pd.read_csv(import_path)

    # Find the closest sim_time
    available_times = df["sim_time"].unique()
    closest_time = available_times[np.abs(available_times - sim_time).argmin()]

    print(f"[Import] Closest recorded sim_time found: {closest_time:.3f}")

    # Filter rows for the closest sim_time
    snapshot_df = df[df["sim_time"] == closest_time]

    if snapshot_df.empty:
        raise ValueError(f"No data found for sim_time = {closest_time:.3f}")

    # Extract particle attributes
    indices = snapshot_df["particle_index"].to_numpy(dtype=np.int32)
    positions = snapshot_df[["x", "y"]].to_numpy(dtype=np.float64)
    velocities = snapshot_df[["vx", "vy"]].to_numpy(dtype=np.float64)
    density = snapshot_df["density"].to_numpy(dtype=np.float64)
    mass = snapshot_df["mass"].to_numpy(dtype=np.float64)
    alpha = snapshot_df["alpha"].to_numpy(dtype=np.float64)
    types = snapshot_df["type"].to_numpy(dtype=np.int32)

    # Sort particles by index to ensure proper order
    sorted_indices = np.argsort(indices)
    positions = positions[sorted_indices]
    velocities = velocities[sorted_indices]
    density = density[sorted_indices]
    mass = mass[sorted_indices]
    alpha = alpha[sorted_indices]
    types = types[sorted_indices]

    # Construct the Particles object
    from dfsph.particles import Particles  # Import only when needed to avoid circular imports

    num_particles = len(indices)
    particles = Particles(num_particles)
    particles.position = positions
    particles.velocity = velocities
    particles.density = density
    particles.mass = mass
    particles.alpha = alpha
    particles.types = types

    print(
        f"[Import] Successfully loaded {num_particles} particles from sim_time = {closest_time:.3f}"
    )

    return particles
