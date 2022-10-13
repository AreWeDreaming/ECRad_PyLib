import numpy as np
from scipy.interpolate import griddata, RectBivariateSpline
    
class IrregularGridAssist():

    def __init__(self, R_points, z_points, quantities, grid_shape = (400,600), log_mask=[]]):
        self.R =  R_points # List of all R from the irregular grid
        self.z =  z_points # List of all R from the irregular grid
        self.quantities = quantities # dictionary of grid quantities
        self.log_mask = log_mask # qauntities in this list will be interpolated logarithmically
        self.grid_shape = grid_shape
        self.grids = {}
        self.spls = {}

    def make_rect_grid(self, names):
        if("R" not in list(self.grids.keys())):
            self.grids["R"] = np.linspace(np.min(self.R), np.max(self.R), self.grid_shape[0])
            self.grids["z"]  = np.linspace(np.min(self.z), np.max(self.z), self.grid_shape[1])
            self.grids["R_mesh"], self.grids["z_mesh"]  = np.meshgrid(self.grids["R"], self.grids["z"], indexing="ij")
        for name in names:
            print("Now gridding " + name)
            if(name in list(self.grids.keys())):
                continue
            # Raw data
            mesh_points = np.array([self.R, self.z]).T
            # Regular grid
            grid_points = np.array([self.grids["R_mesh"].flatten(), self.grids["z_mesh"].flatten()]).T
            if(name in self.log_mask):
                if(np.any(self.quantities[name]) <= 0):
                    raise ValueError(f"Error: {name} contains zeros or negative values unsuitable for logarithmic interpolation!")
                inter_quant = np.log(self.quantities[name])
                fill_value = np.log(np.min(self.quantities[name]))
            else:
                fill_value = 0
            self.grids[name] = griddata(mesh_points, inter_quant, grid_points, \
                                        method="cubic", fill_value=fill_value)
            self.grids[name] = self.grids[name].reshape(self.grid_shape)
            if(name in self.log_mask):
                self.grids[name] = np.exp(self.grids[name])

    def make_rect_spline(self, names):
        for name in names:
            if(name not in self.grids):
                self.make_rect_grided_quantity([name])
            self.spls[name] = RectBivariateSpline(self.grids["R"], self.grids["z"], self.grids[name])