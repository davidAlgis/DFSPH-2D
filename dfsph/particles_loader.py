import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Global cache for simulation times and persistent writer.
_simulation_times_cache = None
_parquet_writer = None
_export_file_path = None


def init_export(export_path):
    """
    Initialize the export binary file for storing simulation snapshots in a
    single Parquet file. Assumes export_path is a file name (or full path).
    If export_path is empty, nothing is done.
    """
    global _parquet_writer, _export_file_path
    if not export_path:
        return

    # Ensure the export file name ends with ".parquet"
    if not export_path.endswith(".parquet"):
        export_path += ".parquet"

    # Create the directory if it doesn't exist.
    folder = os.path.dirname(export_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Remove the file if it already exists.
    if os.path.exists(export_path):
        os.remove(export_path)
    _export_file_path = export_path
    _parquet_writer = None  # We'll create the writer on the first export.
    print(f"[Export] Initialized binary export file: {_export_file_path}")


def export_snapshot(particles, export_path, sim_time):
    """
    Append a snapshot of the simulation's particle data to a single Parquet
file.
    Parameters:
      - particles: The Particles object.
      - export_path: The file name (or full path) to store snapshots.
      - sim_time: The current simulation time.

    This function uses a persistent ParquetWriter.
    
    """
    global _parquet_writer, _export_file_path
    if not export_path:
        return

    # Ensure the file name ends with ".parquet"
    if not export_path.endswith(".parquet"):
        file_path = export_path + ".parquet"
    else:
        file_path = export_path

    # Ensure the folder exists.
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    num_particles = particles.num_particles

    # Prepare the data as a Pandas DataFrame.
    data = {
        "sim_time": np.full(num_particles, sim_time, dtype=np.float64),
        "particle_index": np.arange(num_particles, dtype=np.int32),
        "x": particles.position[:, 0],
        "y": particles.position[:, 1],
        "vx": particles.velocity[:, 0],
        "vy": particles.velocity[:, 1],
        "density": particles.density,
        "mass": particles.mass,
        "alpha": particles.alpha,
        "type": particles.types,
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)

    # Create a ParquetWriter if needed.
    if _parquet_writer is None:
        _parquet_writer = pq.ParquetWriter(
            file_path, table.schema, compression="snappy"
        )
    _parquet_writer.write_table(table)


def close_export():
    """
    Close the persistent Parquet writer if it exists.
    """
    global _parquet_writer
    if _parquet_writer is not None:
        _parquet_writer.close()
        _parquet_writer = None


def _get_simulation_times(export_path):
    """
    Reads and caches available simulation times from the single Parquet file.
    """
    global _simulation_times_cache
    if _simulation_times_cache is None:
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Export file '{export_path}' not found.")
        table = pq.read_table(export_path, columns=["sim_time"])
        _simulation_times_cache = np.sort(table["sim_time"].to_numpy())
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
        return (
            before
            if abs(before - target_time) < abs(after - target_time)
            else after
        )


def import_snapshot(export_path, sim_time, tol=1e-3):
    if not export_path:
        raise ValueError("Export path is empty. Cannot import snapshots.")

    if not export_path.endswith(".parquet"):
        file_path = export_path + ".parquet"
    else:
        file_path = export_path

    available_times = _get_simulation_times(file_path)
    closest_time = _find_closest_time(available_times, sim_time)

    # Instead of filtering for equality, filter for values within tol of
    # closest_time.
    table = pq.read_table(file_path)
    sim_times = table["sim_time"].to_numpy()
    mask = abs(sim_times - closest_time) < tol
    if not mask.any():
        raise ValueError(f"No data found for sim_time = {closest_time:.3f}")
    filtered_table = table.filter(pa.compute.equal(mask, True))

    df = filtered_table.to_pandas()
    if df.empty:
        raise ValueError(f"No data found for sim_time = {closest_time:.3f}")

    # Extract particle attributes.
    indices = df["particle_index"].to_numpy(dtype=np.int32)
    positions = df[["x", "y"]].to_numpy(dtype=np.float64)
    velocities = df[["vx", "vy"]].to_numpy(dtype=np.float64)
    density = df["density"].to_numpy(dtype=np.float64)
    mass = df["mass"].to_numpy(dtype=np.float64)
    alpha = df["alpha"].to_numpy(dtype=np.float64)
    types = df["type"].to_numpy(dtype=np.int32)

    # Sort particles by index.
    sorted_indices = np.argsort(indices)
    positions = positions[sorted_indices]
    velocities = velocities[sorted_indices]
    density = density[sorted_indices]
    mass = mass[sorted_indices]
    alpha = alpha[sorted_indices]
    types = types[sorted_indices]

    # Construct the Particles object.
    from dfsph.particles import \
        Particles  # Import here to avoid circular imports

    num_particles = len(indices)
    particles = Particles(num_particles)
    particles.position = positions
    particles.velocity = velocities
    particles.density = density
    particles.mass = mass
    particles.alpha = alpha
    particles.types = types

    return particles