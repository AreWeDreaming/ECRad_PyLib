'''
Created on June 18, 2019
@author: sdenk
'''
import numpy as np
class ECRHLauncher:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.name = None
        self.R = None
        self.phi = None
        self.error = 0
        self.curv_y = None
        self.curv_z = None
        self.width_y = None
        self.width_z = None
        self.theta_pol = None
        self.phi_tor = None
        self.f = None
        
    def inject_antenna_cpo_values(self, gy):
        self.name = gy.name
        self.R = gy.position.r
        self.phi = gy.position.phi
        self.z = gy.position.z
        self.x = gy.position.r * np.cos(self.phi)
        self.y = gy.position.r * np.sin(self.phi)
        self.error = 0
        self.curv_y = -1.0 / gy.beam.phaseellipse.invcurvrad[0]
        self.curv_z = -1.0 / gy.beam.phaseellipse.invcurvrad[1]
        self.width_y = gy.beam.spot.size[0]
        self.width_z = gy.beam.spot.size[1]
        self.theta_pol,self.phi_tor = make_angles([gy.launchangles.alpha, gy.launchangles.beta], [0,0])
        self.f = gy.frequency
        
    def inject_ECRad_ray_launch(self, ray_launch, ich):
        self.name = ray_launch["diag_name"][ich]
        self.R = ray_launch["R"][ich]
        self.phi = np.deg2rad(ray_launch["phi"][ich])
        self.z = ray_launch["z"][ich]
        self.x = self.R * np.cos(self.phi)
        self.y = self.R * np.sin(self.phi)
        self.error = 0
        self.curv_y = ray_launch["dist_focus"][ich]
        self.curv_z =  self.curv_y 
        self.width_y = ray_launch["beam_width"][ich]
        self.width_z = self.width_y
        self.theta_pol = np.deg2rad(ray_launch["pol_ang"][ich])
        self.phi_tor = np.deg2rad(ray_launch["tor_ang"][ich])
        self.f = ray_launch["f"][ich]
        
        
def make_angles(x, args):
    # ITER -> AUG
    theta_pol = args[0]
    phi_tor = args[1]
    res = np.zeros(2)
    res[0] = np.arcsin(np.cos(x[1]) * np.sin(x[0])) - theta_pol
    res[1] = -np.arctan(np.tan(x[1]) / np.cos(x[0])) - phi_tor
    return res

def invert_angles(theta_pol, phi_tor):
    # AUG -> ITER
    sol = root(make_angles, [theta_pol, phi_tor], args=[theta_pol, phi_tor])
    if(not sol.success):
        print("Failed to convert EC launching angles from ASDEX Upgrade to ITER convention")
        print(sol.message)
        print("Current status", make_angles(sol.x, args=[theta_pol, phi_tor]))
        raise ValueError
    else:
        return sol.x        