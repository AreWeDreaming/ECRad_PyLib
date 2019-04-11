import numpy as np
from plotting_configuration import *

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

if __name__ == "__main__":
    test_wn_PnPoly()
