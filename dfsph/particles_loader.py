import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from concurrent.futures import ThreadPoolExecutor

# Cache for simulation times
_simulation_times_cache = None


def init_export(export_path):
    """
    Initialize the export directory for Parquet snapshots.
    """
    os.makedirs(export_path, exist_ok=True)  # Ensure directory exists
    print(f"[Export] Initialized export directory: {export_path}")


def export_snapshot(particles, export_path, sim_time, num_workers=4):
    """
    Export a snapshot of the simulation's particle data to a Parquet file.

    Parameters:
    - particles: The Particles object.
    - export_path: The directory where Parquet files are stored.
    - sim_time: The current simulation time.
    - num_workers: Number of threads for parallel export.
    """
    num_particles = particles.num_particles
    file_path = os.path.join(export_path, f"snapshot_{sim_time:.5f}.parquet")

    # Prepare the data as a Pandas DataFrame
    data = {
        "particle_index": np.arange(num_particles, dtype=np.int32),
        "x": particles.position[:, 0],
        "y": particles.position[:, 1],
        "vx": particles.velocity[:, 0],
        "vy": particles.velocity[:, 1],
        "density": particles.density,
        "mass": particles.mass,
        "alpha": particles.alpha,
        "type": particles.types
    }

    df = pd.DataFrame(data)
    df.insert(0, "sim_time", sim_time)  # Add sim_time as the first column

    # Write in parallel using PyArrow
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path, compression="snappy")

    print(f"[Export] Snapshot saved: {file_path}")


def _get_simulation_times(export_path):
    """
    Reads and caches available simulation times from Parquet snapshots.
    """
    global _simulation_times_cache
    if _simulation_times_cache is None:
        print(f"[Import] Caching simulation times from {export_path}...")
        files = [
            f for f in os.listdir(export_path) if f.startswith("snapshot_")
        ]
        times = sorted(
            [float(f.split("_")[1].replace(".parquet", "")) for f in files])
        _simulation_times_cache = np.array(times)  # Store as NumPy array
    return _simulation_times_cache


def _find_closest_time(available_times, target_time):
    """
    Uses binary search to quickly find the closest simulation time.
    """
    idx = np.searchsorted(available_times, target_time, side="left")
    if idx == 0:
        return available_times[0]
    elif idx == len(available_times):
        return available_times[-1]
    else:
        before = available_times[idx - 1]
        after = available_times[idx]
        return before if abs(before -
                             target_time) < abs(after - target_time) else after


def import_snapshot(export_path, sim_time):
    """
    Efficiently imports a snapshot of particles from a Parquet file at the closest sim_time.

    Parameters:
    - export_path (str): Directory containing Parquet snapshots.
    - sim_time (float): The target simulation time. The function picks the closest available time.

    Returns:
    - particles (Particles): A `Particles` object reconstructed at the closest `sim_time`.
    """
    print(f"[Import] Loading snapshot for sim_time ≈ {sim_time:.3f}...")

    # Step 1: Get closest available time
    available_times = _get_simulation_times(export_path)
    closest_time = _find_closest_time(available_times, sim_time)

    print(f"[Import] Closest recorded sim_time found: {closest_time:.3f}")

    file_path = os.path.join(export_path,
                             f"snapshot_{closest_time:.5f}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No snapshot file found for time {closest_time}")

    # Step 2: Read the Parquet file
    table = pq.read_table(file_path)
    df = table.to_pandas()

    # Step 3: Extract particle attributes
    indices = df["particle_index"].to_numpy(dtype=np.int32)
    positions = df[["x", "y"]].to_numpy(dtype=np.float64)
    velocities = df[["vx", "vy"]].to_numpy(dtype=np.float64)
    density = df["density"].to_numpy(dtype=np.float64)
    mass = df["mass"].to_numpy(dtype=np.float64)
    alpha = df["alpha"].to_numpy(dtype=np.float64)
    types = df["type"].to_numpy(dtype=np.int32)

    # Step 4: Sort particles by index
    sorted_indices = np.argsort(indices)
    positions = positions[sorted_indices]
    velocities = velocities[sorted_indices]
    density = density[sorted_indices]
    mass = mass[sorted_indices]
    alpha = alpha[sorted_indices]
    types = types[sorted_indices]

    # Step 5: Construct the Particles object
    from dfsph.particles import Particles  # Import here to avoid circular imports
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
