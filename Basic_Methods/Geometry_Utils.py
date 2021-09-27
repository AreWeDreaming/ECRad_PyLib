'''
Created on Sep 7, 2017

@author: sdenk
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import matplotlib.pyplot as plt
from scipy import __version__ as scivers


def rotate_around_axis(N_vec, n_surf, theta=np.pi / 2.e0):
    rotated_N_vec = np.zeros(3)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    ux = np.zeros([3, 3])
    uxu = np.zeros([3, 3])
    R = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if(i == j):
                R[i, j] = cos_theta
            uxu[i, j] = n_surf[i] * n_surf[j]
    ux[0, 1] = -n_surf[2]
    ux[1, 0] = n_surf[2]
    ux[2, 0] = -n_surf[1]
    ux[0, 2] = n_surf[1]
    ux[1, 2] = -n_surf[0]
    ux[2, 1] = n_surf[0]
    for j in range(3):
        R[:, j] += sin_theta * ux[:, j] + (1.e0 - cos_theta) * uxu[:, j]
    print(R)
    for j in range(3):
        rotated_N_vec[j] = np.sum(N_vec[:] * R[j, :])
    return rotated_N_vec

def Snellius_3D(N_vec, n_surf, N_abs_1, N_abs_2):
    ratio = N_abs_1 / N_abs_2
    scalar_k_surf = -np.sum(n_surf * N_vec)
    if(1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2) < 0.e0):
        print("Encountered NAN when making Snell's law")
        print("Refractive index", N_abs_1)
        print("Angle between k and flux surface", np.arccos(-scalar_k_surf) * 180.e0 / np.pi)
        print("Det", 1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2))
    return ratio * N_vec + (ratio * scalar_k_surf - \
                      np.sqrt(1.e0 - ratio ** 2 * (1.e0 - scalar_k_surf ** 2))) * n_surf

def get_Surface_area_of_torus(R, z):  # R and z contour points of the poloidal cross section
    s = np.linspace(0.0, 1.0, len(R))
    dR_spl = InterpolatedUnivariateSpline(s, R).derivative(1)
    dz_spl = InterpolatedUnivariateSpline(s, z).derivative(1)
    IntSpl = InterpolatedUnivariateSpline(s, 2.0 * np.pi * R * np.sqrt(dR_spl(s) ** 2 + dz_spl(s) ** 2))
    return IntSpl.integral(0.0, 1.0)

def get_arclength(R, z):  # R and z contour points of the poloidal cross section
    s = np.linspace(0.0, 1.0, len(R))
    dR_spl = InterpolatedUnivariateSpline(s, R).derivative(1)
    dz_spl = InterpolatedUnivariateSpline(s, z).derivative(1)
    IntSpl = InterpolatedUnivariateSpline(s, np.sqrt(dR_spl(s) ** 2 + dz_spl(s) ** 2))
    return IntSpl.integral(0.0, 1.0)


def get_av_radius(R, z, S, R_ax, z_ax):
    # S is the integrated arc length
    s = np.linspace(0.0, S, len(R))
    r_spline = InterpolatedUnivariateSpline(s, np.sqrt((R - R_ax) ** 2 + (z - z_ax) ** 2))
#    plt.plot(s, r_spline(s))
#    plt.show()
    return r_spline.integral(0.0, S) / S

def eval_z(x, spl):
    if(scivers == '0.12.0'):
        return (spl.ev(x[0], x[1])) ** 2
    else:
        return (spl(x[0], x[1], grid=False)) ** 2

def eval_spline(x, args):
    if(scivers == '0.12.0'):
        return args[0].ev(x[0], x[1])
    else:
        return args[0](x[0], x[1], grid=False)

def maximize_y(x):
    return -x[1] ** 3

def minimize_y(x):
    return x[1] ** 3

def check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z):
    ix1_border = np.where(x > min(x1_cp, x2_cp))[0].flatten()[0]
    iy1_border = np.where(y > min(y1_cp, y2_cp))[0].flatten()[0]
    ix2_border = np.where(x <= max(x1_cp, x2_cp))[0].flatten()[-1]
    iy2_border = np.where(y <= max(y1_cp, y2_cp))[0].flatten()[-1]
    if(ix2_border - ix1_border == -1 and iy2_border - iy1_border == -1):
        # Same cell
        add_point = True
        print("Same cell")
    elif(ix2_border - ix1_border == -1 and iy2_border - iy1_border == 0 and \
         (z[ix1_border, iy1_border] * z[ix2_border, iy1_border] < 0)):
    # Crossing only in y
        add_point = True
        print("y-neighbors")
    elif(iy2_border - iy1_border == -1 and ix2_border - ix1_border == 0 and \
         (z[ix1_border, iy1_border] * z[ix1_border, iy2_border] < 0)):
        # Crossing only in x
        add_point = True
        print("x-neighbors")
    elif(ix2_border - ix1_border == 0 and iy2_border - iy1_border == 0):
        # Crossing cells in either x or y direction -> needs checking
        submat = np.sign(z[ix1_border - 1:ix1_border + 2, iy1_border - 1:iy1_border + 2])
        crossings = 0
        if(submat[0, 1] * submat[1, 1] < 0):
            crossings += 1
        if(submat[1, 0] * submat[1, 1] < 0):
            crossings += 1
        if(submat[1, 1] * submat[1, 2] < 0):
            crossings += 1
        if(submat[1, 1] * submat[2, 1] < 0):
            crossings += 1
        if(crossings > 0 and crossings <= 3):
            # four crossings -> Saddle point
            add_point = True
            print("diagonal neighbors")
        else:
            add_point = False
    else:
        # Multiple cells in between -> do not add
        add_point = False
    return add_point

def get_theta_pol_phi_tor_from_two_points(x1_vec, x2_vec):
    # x should be carthesian coordinates
    # It does not matter if they are rotated around the origin in the torus center
    # by phi. I.e. x=R also works as long as it is consistent.
    # Phi is defined as the angle between the k_1 = -r_1 and k_2 = r_2 - r_1
    R1 = np.linalg.norm(x1_vec[:2])
    R2 = np.linalg.norm(x2_vec[:2])
    theta_pol = get_theta_pol_from_two_points(np.array([R1, x1_vec[2]]), 
                                              np.array([R2, x2_vec[2]]))
    phi_tor = get_phi_tor_from_two_points(x1_vec[:2], x2_vec[:2])
    return theta_pol, phi_tor

def get_theta_pol_from_two_points(R1_vec, R2_vec):
    # R1_vec and R2_vec should be 2D and lie in the R,z plane
    dR = R2_vec[0] - R1_vec[0] 
    dz = R2_vec[1] - R1_vec[1]
    theta_pol = np.rad2deg(np.arctan(dz/dR))
    return theta_pol

def get_phi_tor_from_two_points(x1_vec, x2_vec):
    # x1_vec and x2_vec should be 2D and lie in the x,y plane
    phi_tor = -np.rad2deg(np.arccos((-x1_vec[0] * (x2_vec[0] - x1_vec[0]) - x1_vec[1] * (x2_vec[1] - x1_vec[1])) / 
                                                    (np.linalg.norm(x1_vec) * np.linalg.norm(x2_vec - x1_vec))))
    return phi_tor

class Contouring():
    def __init__(self, x, y, Z, debug=False):
        if(np.any(np.array(Z.shape) < 4)):
            raise ValueError("Need more than 4 entries to find contours")
        self.nx = len(x)
        self.ny = len(y)
        if(Z.shape != (self.nx,self.ny)):
            raise ValueError("x, y improperly shaped")
        self.x = x
        self.y = y
        self.Z = Z
        self.debug=debug
        
    def find_contours(self, h):
        # Contour for single level
        # Sign changes of Z-h are recoreded in pen_points
        # The recorded points are always above or left the penetration point
        # Negative indices indicate a horizontal sign change
        self.pen_points = []
        #horizontal
        # The zeroth x entry can never be a horizontal penetration point
        for i in range(1, self.nx-1):
            for j in range(0, self.ny):
                if((self.Z[i,j] -h) * (self.Z[i + 1,j] - h) < 0.0):
                    self.pen_points.append(-np.array([i,j]))
        # Edge
        for i in range(self.nx-1, self.nx):
            for j in range(0, self.ny):
                if((self.Z[i,j] -h) * (self.Z[i - 1,j] - h) < 0.0):
                    if(np.all(np.sum(np.abs(self.pen_points - (-np.array([i - 1,j]))),axis=1) > 0)):
                        self.pen_points.append(-np.array([i - 1,j]))
        #Vertical
        for i in range(0, self.nx):
            # The zeroth xy entry can never be a vertical penetration point
            for j in range(1, self.ny-1):
                if((self.Z[i,j] - h) * (self.Z[i,j + 1] - h) < 0.0):
                    self.pen_points.append(np.array([i,j]))
        #Edge
        for i in range(0, self.nx):
            for j in range(self.ny-1, self.ny):
                if((self.Z[i,j] - h) * (self.Z[i,j - 1] - h) < 0.0):
                    if(np.all(np.sum(np.abs(self.pen_points - (-np.array([i,j-1]))),axis=1) > 0)):
                        self.pen_points.append(np.array([i,j-1]))
        if(len(self.pen_points) == 0):
            self.contours_found = False
            return
        # Convert the pen_points to complex values which makes the index search much faster
        # Also remove any duplicates we might have picked up
        self.pen_points = list(np.unique(np.array(self.pen_points).T[0] + 1.j * np.array(self.pen_points).T[1]))
        self.contours_found = True
        self.finished_points = []
        self.contour_lines = [[]]
        self.contour_indices = [[]]
        self.contour_closed = []
        next_point = self.pen_points.pop(0)
        Z_spl = RectBivariateSpline(self.x, self.y, self.Z - h)
        N_int = 10
        x_int = np.zeros(N_int)
        y_int = np.zeros(N_int)
        # Open contourlines are reversed to avoid bad starting points
        contour_reversed = False
        # Assemble contours
        while True:
            i_next = int(np.real(next_point))
            j_next = int(np.imag(next_point))
            if(i_next < 0 or j_next < 0):
                # Horiontal penetration point
                x_int[:] = np.linspace(self.x[-i_next], self.x[-i_next + 1], N_int)
                y_int[:] = self.y[-j_next]
                Z_int = Z_spl(x_int, y_int, grid=False)
                root_spl = InterpolatedUnivariateSpline(x_int, Z_int)
                roots = root_spl.roots()
                if(len(roots) == 0):
                    print("Found no roots for this penetration point")
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.contour(self.x, self.y, self.Z.T - h, levels=[0.0])
                    ax.vlines(self.x, np.min(self.y), np.max(self.y), linewidths=0.2)
                    ax.hlines(self.y, np.min(self.x), np.max(self.x), linewidths=0.2)
                    ax.plot(x_int, y_int, "r--", linewidth=4)
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)
                    ax2.plot(x_int, Z_int)
                    plt.show()
                    raise ValueError("Found  no roots for this penetration point")
                for root in roots:
                    self.contour_lines[-1].append([root, y_int[0]])
            else:
                # Horiontal penetration point
                x_int[:] = self.x[i_next]
                y_int[:] = np.linspace(self.y[j_next], self.y[j_next + 1], N_int)
                Z_int = Z_spl(x_int, y_int, grid=False)
                root_spl = InterpolatedUnivariateSpline(y_int, Z_int)
                roots = root_spl.roots()
                if(len(roots) == 0):
                    print("Found no roots for this penetration point")
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.contour(self.x, self.y, self.Z.T - h, levels=[0.0])
                    ax.vlines(self.x, np.min(self.y), np.max(self.y), linewidths=0.2)
                    ax.hlines(self.y, np.min(self.x), np.max(self.x), linewidths=0.2)
                    ax.plot(x_int, y_int, "r--", linewidth=4)
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)
                    ax2.plot(y_int, Z_int)
                    plt.show()
                    raise ValueError("Found no roots for this penetration point")
                for root in roots:
                    self.contour_lines[-1].append([x_int[0], root])
            self.contour_indices[-1].append(next_point)
            self.finished_points.append(next_point)
            # Find the next point
            found_next, isclosed, next_point = self._find_next(next_point)
            if(not found_next):
                if(isclosed and len(self.contour_lines[-1]) > 2):
                    # Close the contour
                    self.contour_lines[-1].append(self.contour_lines[-1][0])
                    self.contour_indices[-1].append(self.contour_indices[-1][0])
                elif(not contour_reversed):
                    self.contour_lines[-1] = self.contour_lines[-1][::-1]
                    self.contour_indices[-1] = self.contour_indices[-1][::-1]
                    contour_reversed = True
                    found_next, isclosed, next_point = self._find_next(self.contour_indices[-1][-1])
                if(not found_next):
                    contour_reversed = False
                    self.contour_closed.append(isclosed)
                    self.contour_lines[-1] = np.array(self.contour_lines[-1])
                    self.contour_indices[-1] = np.array(self.contour_indices[-1])
                    if(len(self.pen_points) == 0):
                        break
                    next_point = self.pen_points.pop(0)
                    self.contour_lines.append([])
                    self.contour_indices.append([])
                            
    def _find_next(self, point):
        hor_index_inc = 1.0
        ver_index_inc = 1.j
        if(np.real(point) < 0 or np.imag(point) < 0):
            # Two horizontal neighbours (negative)
            # and four vertical neighbours (positive)
            candidates = np.array([-point, -point + hor_index_inc, \
                                   point + ver_index_inc , \
                                   point - ver_index_inc, \
                                   -point - ver_index_inc + hor_index_inc, \
                                   -point - ver_index_inc])
            if(np.imag(point) == 0):
                candidates = candidates[0:3]
        else:
            # Four horizontal neighbours (negative)
            # and two vertical neighbours (positive(
            candidates = np.array( [point - hor_index_inc, \
                                    point + hor_index_inc , \
                                    -point + hor_index_inc, \
                                    -point, -point - ver_index_inc, \
                                    -point - ver_index_inc + hor_index_inc])
            if(np.real(point) == 0):
                candidates = candidates[1:]
            if(np.imag(point) == 0):
                candidates = candidates[0:4]
        candidates = candidates[np.abs(np.real(candidates)) < len(self.x)]
        candidates = candidates[np.abs(np.imag(candidates)) < len(self.y)]
        if(len(self.pen_points) > 0):
            # This is also called a last time to check if the last contour is really closed
            for candidate in candidates:
                if(np.any(np.abs(np.real(self.pen_points - candidate)) + 
                          np.abs(np.imag(self.pen_points - candidate)) == 0.0)):
                        return True, False, self.pen_points.pop(self.pen_points.index(candidate))
        for candidate in candidates:
            if(np.abs(np.real(self.contour_indices[-1][0] - candidate)) + \
               np.abs(np.imag(self.contour_indices[-1][0] - candidate)) == 0.0):
                    return False, True, None
        if(self.debug):
            plt.plot(self.x[np.abs(np.int(np.real(point)))], self.y[np.abs(np.int(np.imag(point)))], "o")
            for candidate in candidates:
                plt.plot(self.x[np.abs(np.int(np.real(candidate)))], self.y[np.abs(np.int(np.imag(candidate)))], "+r")
            plt.show()
        return False, False, None

def get_contour(x, y, z_in, val):
    # Only works if z has exactly a single, convex, nested and closed contour
    # Easiest if we look for the zero contour
    z = np.copy(z_in) - val
    # Now find the largest and smallest y with a sign change in y-direction
    iy_min = len(y)
    iy_max = 0
    for ix in range(len(x)):
        for isc in np.where(z[ix, 1:] - z[ix, 0:len(y) - 1])[0]:
            if(isc < iy_min):
                iy_min = isc
            if(isc > iy_max):
                iy_max = isc
    if(iy_min - 1 > 0):
        iy_min -= 1
    if(iy_max + 1 < len(y)):
        iy_max += 1
    spl = RectBivariateSpline(x, y, z)
    dxmin = np.min(np.abs(x[1:] - x[0:len(x) - 1]))  # smallest spacing in x
    dymin = np.min(np.abs(y[1:] - y[0:len(y) - 1]))  # smallest spacing in y
    while(y[iy_max] - y[iy_min] < 50 * dymin):
        dymin *= 2
    x_grid = np.arange(np.min(x), np.max(x), dxmin)
    y_grid = np.arange(y[iy_min], y[iy_max], dymin)
    # Assures that contours are less than one grid cell apart at worst
    y_inter = np.zeros(len(x_grid))
    cont = []
    for i in range(len(y_grid)):
        y_inter[:] = y_grid[i]
        z_inter = spl(x_grid, y_inter, grid=False)
        roots = InterpolatedUnivariateSpline(x_grid, z_inter).roots()
        for root in roots:
            cont.append([root, y_grid[i]])
    cont = np.array(cont)
    x_geo = np.mean(cont.T[0])
    y_geo = np.mean(cont.T[1])
    thetas = np.arctan2(cont.T[1] - y_geo, cont.T[0] - x_geo)
    isort = np.argsort(thetas)
    i_last = isort[0]
    i_start = i_last  # Starting point of current contour
    cur_cont = [[cont.T[0][i_last], cont.T[1][i_last]]]
    sorted_conts = []
    insort = 1
    finished = np.zeros(len(isort), dtype=np.bool)
    finished[0] = True
    move_direction = 1  # Reversed if looking for further points of open contour
    while False in finished:
        if(not finished[insort]):
            i = isort[insort]
            x1_cp = cont.T[0][i_last]
            x2_cp = cont.T[0][i]
            y1_cp = cont.T[1][i_last]
            y2_cp = cont.T[1][i]
            if(check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z)):
#                plt.plot(cont.T[0], cont.T[1], "+")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "-")
#                plt.show()
                cur_cont.append([x2_cp, y2_cp])
                finished[insort] = True
                i_last = i
#            else:
#                plt.plot([x[ix1_border - 1], x[ix1_border], x[ix1_border + 1], \
#                          x[ix1_border - 1], x[ix1_border], x[ix1_border + 1], \
#                          x[ix1_border - 1], x[ix1_border], x[ix1_border + 1]], \
#                         [y[iy1_border - 1], y[iy1_border - 1], y[iy1_border - 1], \
#                          y[iy1_border], y[iy1_border], y[iy1_border], \
#                          y[iy1_border + 1], y[iy1_border + 1], y[iy1_border + 1]], "+")
# #                plt.plot(cont.T[0], cont.T[1], "+")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "-")
#                plt.plot([x1_cp, x2_cp], [y1_cp, y2_cp], "*")
#                plt.show()
        insort += move_direction
        if(insort == len(isort) and not np.all(finished)):
        # End of current contour reached
        # First check if start point of last contour can be continued
            i_last = i_start
            move_direction = -1
            insort += move_direction
        if(insort < 0  and not np.all(finished)):
        # Current contour finished
            sorted_conts.append(cur_cont)
        # -> start a new one
            cur_cont = []
            insort = np.where(np.logical_not(finished))[0][0]
            i_last = isort[insort]
            finished[insort] = True
            cur_cont.append([cont.T[0][i_last], cont.T[1][i_last]])
            if(len(isort[np.logical_not(finished)]) > 1):
                insort = np.where(np.logical_not(finished))[0][0]
            move_direction = +1
    # Finally go through all contours and check for closed ones
    closed_info = np.zeros(len(sorted_conts), dtype=np.bool)
    for i_cont in range(len(sorted_conts)):
        cont = sorted_conts[i_cont]
        i = isort[insort]
        x1_cp = cont[0][0]
        x2_cp = cont[0][1]
        y1_cp = cont[-1][0]
        y2_cp = cont[-1][1]
        # Build connection between points
        if(check_neighbor(x1_cp, y1_cp, x2_cp, y2_cp, x, y, z)):
        # Closed contour append first point at end
            sorted_conts[i_cont].append([x1_cp, y1_cp])
            closed_info[i_cont] = True
        # Convert all contours to np arrays
        sorted_conts[i_cont] = np.array(sorted_conts[i_cont])
        plt.plot(sorted_conts[i_cont].T[0], sorted_conts[i_cont].T[1], "-")
#    x_cont = np.concatenate([cont.T[0][isort], [cont.T[0][np.argmin(thetas)]]])
#    y_cont = np.concatenate([cont.T[1][isort], [cont.T[1][np.argmin(thetas)]]])
#    plt.plot(x_cont, y_cont, "-")
    plt.show()
    return closed_info, sorted_conts

def isLeft(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - \
                    (P2[0] - P0[0]) * (P1[1] - P0[1])

def wn_PnPoly(x, poly):
    # Closed polynom poly!
    wn = 0
    for i in range(len(poly) - 1):
        if(poly[i][1] <= x[1]):
            if(poly[i + 1][1] > x[1]):
                if (isLeft(poly[i], poly[i + 1], x) > 0):
                    wn += 1
        else:
            if(poly[i + 1][1] <= x[1]):
                if (isLeft(poly[i], poly[i + 1], x) < 0):
                    wn -= 1
    return wn

def test_wn_PnPoly():
    # Check whether point inside or outside polynome using the winding number test
    # Algorithm taken from http://geomalgorithms.com/a03-_inclusion.html
    wall = np.loadtxt("../ECRad_Pylib/ASDEX_Upgrade_vessel.txt")
    poly = []
    for i in range(len(wall.T[0])):
        poly.append([wall.T[0, i], wall.T[1, i]])
    poly = np.array(poly)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wall.T[0], wall.T[1], "-k")
    x = np.random.rand(100)
    x = 1.23 + x * 0.7
    y = np.random.rand(100)
    y = -0.55 + y * 1.33
    for i in range(100):
        if(wn_PnPoly(np.array([x[i], y[i]]), poly) != 0):
            ax.plot(x[i], y[i], "+g")
        else:
            ax.plot(x[i], y[i], "+r")
    x = np.random.rand(500)
    x = x * 3.0
    y = np.random.rand(500)
    y = (0.5 - y) * 3.0
    for i in range(500):
        if((x[i] < 1.0 or x[i] > 2.27) or \
           (y[i] < -1.27 or y[i] > 1.27)):
            if(wn_PnPoly(np.array([x[i], y[i]]), poly) == 0):
                ax.plot(x[i], y[i], "*g")
            else:
                ax.plot(x[i], y[i], "*r")
    plt.show()


if(__name__ == "__main__"):
    x = np.linspace(1.0, 2.0, 500)
    y = np.linspace(-1.0, 1.0, 500)
    y_mesh, x_mesh = np.meshgrid(y,x)
    Z = np.sqrt((x_mesh - 1.5)**2 + y_mesh**2)
    cont_maker = Contouring(x,y,Z, False)
    h = 0.55
#     plt.contour(x, y, Z.T, levels=[h])
    plt.vlines(x, np.min(y), np.max(y), linewidths=0.2)
    plt.hlines(y, np.min(x), np.max(x), linewidths=0.2)
    cont_maker.find_contours(h)
    for cont, closed in zip(cont_maker.contour_lines, cont_maker.contour_closed):
        print(closed)
        plt.plot(cont.T[0], cont.T[1], "+", linestyle="-")
    plt.show()
#     n_surf = np.array([-0.158822604797721, 0.976942169466021, -0.142686291297696])
#     n_surf /= np.sqrt(np.sum(n_surf ** 2))
#     vec_in = np.array([0.4, 0.2, 0.8])
#     print(Snellius_3D(vec_in, n_surf, 1.0, 0.88))


#    print("Norm before", np.sqrt(np.sum(vec_in ** 2)))
#    vec_out = rotate_around_axis(vec_in, n_surf, theta=np.pi / 2.784e0)
#    print(vec_out, np.sqrt(np.sum(vec_out ** 2)))
