class DFSPHInitConfig:

    def __init__(self,
                 h,
                 dt,
                 grid_origin,
                 grid_size,
                 cell_size,
                 rest_density=1027,
                 water_viscosity=0.01,
                 surface_tension_coeff=2.0):
        self.h = h
        self.dt = dt
        self.grid_origin = grid_origin
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.rest_density = rest_density
        self.water_viscosity = water_viscosity
        self.surface_tension_coeff = surface_tension_coeff
