# #-- General Modules--##
from GlobalSettings import *
import numpy as np
from plotting_configuration import *
from netCDF4 import Dataset
from equilibrium_utils_AUG import EQData
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.markeredgewidth'] = 1

__all__ = ['bmw']

class bmw():

    '''
     Routines that work with the BMW and EXTENDER codes.
        - read_bmw:            Reads BMW netCDF output.
        - read_extender:       Reads EXTENDER netCDF output.
        - write_pfm:           Writes the PFM to the BMW netCDF file.
        - get_separatrix:      Computes the 3D separatrix from the PFM.
        - curl_potential:      Takes the curl operator in cylindrical coordinates.
        - bmw_divergence:      Takes the divergence operator in cylindrical coordinates.
        - bmw_to_emc3:         Writes the BMW fields in EMC3-EIRENE format.
        - bmw_to_gourdon:      Writes the output field from BMW in 'MFBE'-like format for the GOURDON code.
        - extender_to_gourdon: Writes the output field from EXTENDER in 'MFBE'-like format for the GOURDON code.
    '''


    def read_bmw(self, shot, time, eq_exp, eq_diag, eq_ed, path_bmw, file_bmw):

        '''
        ----------------------------Description-----------------------------------
         Reads and plots the output from the BMW code
        -------------------------------Input--------------------------------------
        -------------------------------Output-------------------------------------
        ------------------------------Comments------------------------------------
         BMW outputs in netCDF
         Since the BMW magnetic potentials and fields are evaluated at the 
         unprimed grid (integer grid):
         phi_min = 0.
         phi_max = (2.*np.pi / nfp) * (nphi-1)/(float(nphi))
         Since the currents are evalauted at the primed-grid (half-integer grid),
         {x,y,z} are coordinates of the primed-grid.
         The unprimed grid is always interleaved with the primed grid and both
         have the same number of toroidal planes, which is independent of the
         VMEC toroidal planes. The VMEC toroidal planes determine the resolution/quality
         of the harmonics. Then, this harmonics are used to reconstruct the currents
         in an arbitatry number of toroidal planes in the primed grid.
        --------------------------------------------------------------------------
        '''
        print("Reading initial Cliste equilibrium")
        EQ_obj = EQData(shot, EQ_exp=eq_exp, EQ_diag=eq_diag, EQ_ed=eq_ed)
        EQSlice = EQ_obj.read_EQ_from_shotfile(time)
        print('   - Reading BMW output file...')
        self.path_bmw = path_bmw
        self.file_bmw = file_bmw

        trial = Dataset(path_bmw + file_bmw, mode='r')
        self.trial = trial

        nr = int(str(trial.dimensions['r']).split()[-1])
        nz = int(str(trial.dimensions['z']).split()[-1])
        nphi = int(str(trial.dimensions['phi']).split()[-1])
        nfp = trial.variables['nfp'][:]
        rmin = trial.variables['rmin'][:]
        rmax = trial.variables['rmax'][:]
        zmin = trial.variables['zmin'][:]
        zmax = trial.variables['zmax'][:]
        ap_grid = trial.variables['ap_grid'][:]
        br_grid = trial.variables['br_grid'][:]
        bp_grid = trial.variables['bp_grid'][:]
        bz_grid = trial.variables['bz_grid'][:]

        r_extent = np.linspace(rmin, rmax, nr)
        z_extent = np.linspace(zmin, zmax, nz)
        R_mesh, Z_mesh = np.meshgrid(r_extent, z_extent)

        # Computes the PFM = 2 * pi * R * Aphi
        pfm_grid = np.zeros_like(ap_grid)
        for pp in range(0, nphi):
            pfm_grid[pp, :, :] = 2.*np.pi * ap_grid[pp, :, :] * R_mesh
        self.pfm_grid = np.sqrt((pfm_grid - EQSlice.Psi_ax) / (EQSlice.Psi_sep - EQSlice.Psi_ax))



        # Calculate quantities in phi-direction
        delt_phi = 2.*np.pi / (nfp * nphi)
        phi_min_upgrid = 0.
        phi_max_upgrid = (2.*np.pi / nfp) * (nphi - 1) / (float(nphi))
        phi_mesh_upgrid = np.linspace(phi_min_upgrid, phi_max_upgrid, nphi)

        self.nfp = nfp
        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax
        self.r_extent = r_extent
        self.z_extent = z_extent
        self.R_mesh = R_mesh
        self.Z_mesh = Z_mesh
        self.delt_phi = delt_phi
        self.phi_min_upgrid = phi_min_upgrid
        self.phi_max_upgrid = phi_max_upgrid
        self.phi_mesh_upgrid = phi_mesh_upgrid
        self.br_grid = br_grid
        self.bp_grid = bp_grid
        self.bz_grid = bz_grid


if(__name__ == "__main__"):
    bmw_obj = bmw()
    bmw_obj.read_bmw(34632, 5.0, "MICDU", "EQB", 13, "/tokp/work/sdenk/BMW/", "bmw_free3d_34632_99999_v8_m2.2_n8_m24_256_512_180_1001.nc")
    plt.plot(bmw_obj.R_mesh.T.flatten(), bmw_obj.Z_mesh.T.flatten(), ".")
#    plt.figure()
#    plt.plot(bmw_obj.z_extent)
#    plt.figure()
#    plt.plot(bmw_obj.phi_mesh_upgrid)
#    print(bmw_obj.r_extent.shape, bmw_obj.z_extent.shape, bmw_obj.phi_mesh_upgrid.shape, bmw_obj.pfm_grid.shape, bmw_obj.pfm_grid[0].shape)
#    plt.contour(bmw_obj.r_extent, bmw_obj.z_extent, bmw_obj.pfm_grid[0], levels=[1.006])
#    print(bmw_obj.br_grid.shape)
    plt.show()
