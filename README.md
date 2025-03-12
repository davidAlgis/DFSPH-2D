# DFSPH-2D

A Python implementation of a 2D divergence-free SPH (DFSPH) solver. This solver is based on the method described by Koschier and Bender (2017) for fast incompressible flow simulation.

![](captures/dfsph_test.gif)

> [!WARNING]  
> The gif above feels like it's a real time computation. Unfortunately it's an acceleration of the capture of a simulation, which takes likes 2 minutes to be computed.

## Installation

Ensure you have Python 3.8+ installed. Then, install the package in editable mode using the provided `pyproject.toml`:

```bash
pip install -e .
```

## Usage

Run the simulation via the main entry point:

```bash
python main.py [options]
```

You can customize parameters using the following options:

- **-r, --support_radius**  
  SPH support radius (default: 0.2742/2).

- **-dt, --timestep**  
  Time step for the simulation (default: 0.01).

- **-s, --steps**  
  Number of simulation steps (default: 1000). *(Used in headless mode.)*

### Box & Grid Parameters

- **--box_origin**  
  Origin of the box for particle initialization (default: [-0.8, -0.8]).

- **--box_size**  
  Size of the box for particle initialization (default: [1.4, 1.4]).

- **--grid_size**  
  Grid dimensions as (width, height) (default: [2, 2]).

- **--grid_origin**  
  Position of the grid in simulation space (default: [-1, -1]).

### Physical Parameters

- **--rest_density**  
  Rest density of the fluid (default: 1027.0).

### Visualization

- **-v, --visualize**  
  Enable real-time visualization (default: enabled).

### Export/Import

- **-e, --export_results**  
  File name to export particle data (default: empty).

- **-i, --import_results**  
  File name to import particle data for visualization (default: empty).

- **-ii, --import_init**  
  File name to import initial particle configuration (default: empty).

## Reference

Koschier, M., & Bender, J. (2017). *DFSPH: Fast Divergence-Free SPH for Incompressible Flows*. [Link to paper](https://doi.org/10.1109/TOG.2017.2709662)
