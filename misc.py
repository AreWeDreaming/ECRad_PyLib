'''
Created on Apr 24, 2018

@author: sdenk
'''

import numpy as np
from plotting_configuration import *
from mpl_toolkits.mplot3d import Axes3D

def illustrate_solid_angle():
    a = np.array([1.0, 0, 0])
    b = np.array([0, 1.0, 0])
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    alpha = np.arctan2(c[1], c[0])
    beta = np.arccos(c[2])
    fig = plt.figure(figsize=(12.5, 8.5))
    ax = fig.gca(projection='3d')
    x = []
    y = []
    z = []
    for s in np.linspace(-0.5, 0.5, 20):
        x.append([-a[0] + s * b[0], a[0] + s * b[0]])
        y.append([-a[1] + s * b[1], a[1] + s * b[1]])
        z.append([-a[2] + s * b[2], a[2] + s * b[2]])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    nx = np.array([0.0, np.cos(alpha) * np.sin(beta)])
    ny = np.array([0.0, np.sin(alpha) * np.sin(beta)])
    nz = np.array([0.0, np.cos(beta)])
    ax.plot(nx, ny, nz, "-r")
    for phi in np.linspace(alpha - np.pi / 2, alpha + np.pi / 2, 10):
        for theta in np.linspace(beta - np.pi / 2, beta + np.pi / 2, 10):
            kx = np.array([0.0, np.cos(phi) * np.sin(theta)])
            ky = np.array([0.0, np.sin(phi) * np.sin(theta)])
            kz = np.array([0.0, np.cos(theta)])
            ax.plot(kx, ky, kz, "-b")
    ax.plot_surface(x, y, z)
    plt.show()


if(__name__ == "__main__"):
    illustrate_solid_angle()
