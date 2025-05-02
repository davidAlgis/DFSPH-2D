"""
grid.py  – spatial hash grid for DF‑SPH (Numba‑ready, no nested containers)

Public interface (unchanged)
----------------------------
    Grid(grid_origin, grid_size, cell_size)
    Grid.find_neighbors(particles, search_radius)
"""

import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from numba import njit, prange, types
from numba.typed import Dict

# -------------------------------------------------------------------------
# 1.  Helpers: 2‑D cell coordinate  ⇨  single 64‑bit key
# -------------------------------------------------------------------------


@njit(inline="always")
def _cell_key(cx: int, cy: int) -> np.int64:
    """Pack (cx, cy) into one signed 64‑bit integer."""
    return (np.int64(cx) << 32) ^ np.int64(cy & 0xFFFFFFFF)


# -------------------------------------------------------------------------
# 2.  Cell index computation
# -------------------------------------------------------------------------


@njit
def compute_cell_indices_numba(
    positions: np.ndarray, grid_origin: np.ndarray, inv_cell_size: float
) -> np.ndarray:
    """
    Vectorised → no Python loop.
    """
    rel = (positions - grid_origin) * inv_cell_size
    return np.floor(rel).astype(np.int64)  # (n, 2)


# -------------------------------------------------------------------------
# 3.  Build a *linked‑list* hash grid
# -------------------------------------------------------------------------


@njit
def _build_linked_grid(cell_indices: np.ndarray):
    """
    Returns
    -------
    head      : Dict[int64 → int64]   (cell_key → first particle in cell)
    next_part : np.ndarray[int64]     (linked list inside each cell)
    """
    n = cell_indices.shape[0]
    next_part = np.full(n, -1, dtype=np.int64)

    head = Dict.empty(
        key_type=types.int64, value_type=types.int64
    )  # -1 = empty

    for i in range(n):
        key = _cell_key(cell_indices[i, 0], cell_indices[i, 1])
        if key in head:
            next_part[i] = head[key]
        head[key] = i  # insert at head (O(1))

    return head, next_part  # both live in native memory


# -------------------------------------------------------------------------
# 4.  Two‑pass neighbour search (count, then fill)
# -------------------------------------------------------------------------


@njit(parallel=True)
def find_neighbors_hashgrid_numba(
    positions: np.ndarray, cell_indices: np.ndarray, h: float, cell_size: float
):
    """
    Returns the usual SoA neighbour layout:
        neighbor_indices  (total,) int64
        neighbor_counts   (n,)     int64
        neighbor_starts   (n,)     int64
    """
    n = positions.shape[0]
    h2 = h * h
    r = int(math.ceil(h / cell_size))  # cells to either side

    head, next_part = _build_linked_grid(cell_indices)

    # ----------------------------------- pass 1 : counts
    neighbor_counts = np.zeros(n, dtype=np.int64)

    for i in prange(n):
        cx, cy = cell_indices[i, 0], cell_indices[i, 1]
        cnt = 0

        for dx in range(-r, r + 1):
            nx = cx + dx
            for dy in range(-r, r + 1):
                ny = cy + dy
                key = _cell_key(nx, ny)
                if key in head:
                    j = head[key]
                    while j != -1:
                        if i != j:
                            dxp = positions[i, 0] - positions[j, 0]
                            dyp = positions[i, 1] - positions[j, 1]
                            if dxp * dxp + dyp * dyp < h2:
                                cnt += 1
                        j = next_part[j]
        neighbor_counts[i] = cnt

    # exclusive prefix sum → neighbor_starts
    neighbor_starts = np.empty(n, dtype=np.int64)
    total = 0
    for i in range(n):
        neighbor_starts[i] = total
        total += neighbor_counts[i]

    neighbor_indices = np.empty(total, dtype=np.int64)

    # ----------------------------------- pass 2 : fill indices
    for i in prange(n):
        cx, cy = cell_indices[i, 0], cell_indices[i, 1]
        write = neighbor_starts[i]

        for dx in range(-r, r + 1):
            nx = cx + dx
            for dy in range(-r, r + 1):
                ny = cy + dy
                key = _cell_key(nx, ny)
                if key in head:
                    j = head[key]
                    while j != -1:
                        if i != j:
                            dxp = positions[i, 0] - positions[j, 0]
                            dyp = positions[i, 1] - positions[j, 1]
                            if dxp * dxp + dyp * dyp < h2:
                                neighbor_indices[write] = j
                                write += 1
                        j = next_part[j]

    return neighbor_indices, neighbor_counts, neighbor_starts


# -------------------------------------------------------------------------
# 5.  OOP wrapper – public API identical to the old Grid class
# -------------------------------------------------------------------------


class Grid:
    """
    Uniform hash‑grid – now with O(n) neighbour queries.
    """

    def __init__(
        self,
        grid_origin: Tuple[float, float],
        grid_size: Tuple[float, float],
        cell_size: float,
    ):
        self.grid_origin = np.asarray(grid_origin, dtype=np.float64)
        self.grid_size = np.asarray(grid_size, dtype=np.float64)
        self.grid_end = self.grid_origin + self.grid_size

        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size

        # kept for legacy debug / visualisation tools
        self.cells = defaultdict(list)

    # ---------- debug helper ------------------------------------------------
    def _compute_cell_index(self, position: np.ndarray) -> Tuple[int, int]:
        idx = np.floor((position - self.grid_origin) * self.inv_cell_size)
        return int(idx[0]), int(idx[1])

    def insert_particles(self, particles):
        """(Optional) refresh Python‑side cell map for debug rendering."""
        self.cells.clear()
        for i in range(particles.num_particles):
            cell_idx = self._compute_cell_index(particles.position[i])
            self.cells[cell_idx].append(i)

    # ---------- main entry point -------------------------------------------
    def find_neighbors(
        self, particles, search_radius: float
    ) -> List[List[int]]:
        """
        Fills particles.neighbor_* arrays and returns a list‑of‑lists for
        any legacy code that still expects that structure.
        """
        pos = np.ascontiguousarray(particles.position, dtype=np.float64)

        cell_indices = compute_cell_indices_numba(
            pos, self.grid_origin, self.inv_cell_size
        )

        n_inds, n_cnts, n_starts = find_neighbors_hashgrid_numba(
            pos, cell_indices, search_radius, self.cell_size
        )

        # Python list‑of‑lists (slow but only used for debug / export)
        neighbors_py: List[List[int]] = []
        for i in range(particles.num_particles):
            s = n_starts[i]
            c = n_cnts[i]
            neighbors_py.append(list(n_inds[s : s + c]))

        # publish results
        particles.neighbors = neighbors_py
        particles.neighbor_indices = n_inds
        particles.neighbor_counts = n_cnts
        particles.neighbor_starts = n_starts

        return neighbors_py
