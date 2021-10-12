import numpy as np                # trig functs
import scipy.constants as const   # to get physical constants
import matplotlib.pyplot as plt   
import matplotlib.patches as pat  # to draw rectangles
from Basic_Methods.Geometry_Utils import get_theta_pol_from_two_points, get_phi_tor_from_two_points

def get_ECE_launch_v2(wgIn, antenna, dtoECESI, freqsSI, dfreqsSI):
    ECE_launch = {}
    ECE_launch["f"] = freqsSI
    ECE_launch["df"] = dfreqsSI
    ECE_launch["pol_coeff_X"] = -np.ones(ECE_launch["f"].shape)
    for key in ["R", 'phi', "z", "theta_pol", "phi_tor", "dist_focus", "width"]:
        ECE_launch[key] = np.zeros(ECE_launch["f"].shape)
    # Middle between sector 8 and 9 and 22.5 degrees/sector
    ECE_launch["phi"][:] = 22.5 * 8.5
    # Use a position outside the flux matrix and beyond the window
    R_start = 3.540
    # Position at center of machine for second point
    R_geo = 1.65
    for i, f in enumerate(freqsSI):
        x_tor, y  = plot1DECE(wgIn, antenna, dtoECESI * 1.e3, f/1.e9, "toroidal", doPlot=False, verb=False)
        y = y.T
        x_pol, z = plot1DECE(wgIn, antenna, dtoECESI * 1.e3, f/1.e9, "poloidal", doPlot=False, verb=False)
        z = z.T
        i_start_ax_tor = np.argmin(np.abs(x_tor - R_start))
        i_start_ax_pol = np.argmin(np.abs(x_pol - R_start))
        i_geo_ax_tor = np.argmin(np.abs(x_tor - R_geo))
        i_geo_ax_pol = np.argmin(np.abs(x_pol - R_geo))
        x1_mid_tor = np.array([x_tor[i_start_ax_tor], y[2][i_start_ax_tor]])
        x1_mid_pol = np.array([x_pol[i_start_ax_pol], z[2][i_start_ax_pol]])
        x2_mid_tor = np.array([x_tor[i_geo_ax_tor], y[2][i_geo_ax_tor]])
        x2_mid_pol = np.array([x_pol[i_geo_ax_pol], z[2][i_geo_ax_pol]])
        ECE_launch["R"][i] = x1_mid_pol[0]
        ECE_launch["phi"][i] += np.rad2deg(np.arctan2(x1_mid_tor[1], x1_mid_tor[0]))
        ECE_launch["z"][i] = x1_mid_pol[1]
        # The synthetic CECE diagnostic does not care about the toroidal plane
        # Therefore, we throw it out
        ECE_launch["width"][i] = np.abs(z[0][i_start_ax_pol] - z[4][i_start_ax_pol])
        ECE_launch["theta_pol"][i] = \
                get_theta_pol_from_two_points(x1_mid_pol, x2_mid_pol)
        ECE_launch["phi_tor"][i] = \
                get_phi_tor_from_two_points(x1_mid_tor, x2_mid_tor)
        # Calculate intersections in pol and tor plane with central ray for both peripheral rays
        x_mid_vec_tor = np.array([x_tor[i_geo_ax_tor], y[1][i_geo_ax_tor]]) - x1_mid_tor
        x_mid_vec_pol = np.array([x_pol[i_geo_ax_pol], z[1][i_geo_ax_pol]]) - x1_mid_pol
        x_intersects = []
        y_intersects = []
        z_intersects = []
        for i_periph in [0,4]:
            # Compute intersections between central and the two peripheral rays
            # Need to select the correct indices of the central vector, hence the masks
            # First toroidal plane then poloidal plane
            x_intersects.append([])
            for x, x0, x_vec, ax, i_start, i_centre, ax_intersects in zip([x_tor, x_pol], [x1_mid_tor, x1_mid_pol], 
                                                  [x_mid_vec_tor, x_mid_vec_pol], [y, z],
                                                  [i_start_ax_tor, i_start_ax_pol], [i_geo_ax_tor, i_geo_ax_pol], \
                                                  [y_intersects, z_intersects]):
                x0_periph = np.array([x[i_start], ax[i_periph][i_start]])
                k_periph = np.array([x[i_centre], ax[i_periph][i_centre]]) - x0_periph
                # R.H.S. of the matrix equation
                b = x0 - x0_periph
                #      ( -x_mid_vec[0] k_periph[0] )
                # A = (                             ) 
                #      ( -x_mid_vec[1] k_periph[1] )
                A = np.array([[-x_vec[0], k_periph[0]],
                              [-x_vec[1], k_periph[1]]])
                s1, s2 = np.linalg.solve(A, b)
                x_intersects[-1].append(x0[0] + x_vec[0]* s1)
                ax_intersects.append(x0[1] + x_vec[1] * s1)
        # ECRad needs the real distance between origin and focus point. Since the optical axis is not aligned with R
        # we need to also consider the distance in the y and z plane when calculating the distance between focus and
        # origin
        # Due to numerical errors the R is not precisely the same for the toroidal and poloidal projections
        # Hence we average over the two different Rs to get a single value
        x_orig_3D = np.array([(x1_mid_tor[0] + x1_mid_pol[0])/2.0, x1_mid_tor[1], x1_mid_pol[1]])
        # The beam is astigmatic and the poloidal and toroidal plane have different focus points in x or R
        # We first average over the different focus points in R for the two different projections
        # This is necessary so we can establish a single focus point in 3D for each peripheral ray
        # We discard the one from the toroidal plane because the synthetic diagnostic for GENE does not care about the toroidal plane
        x_intersects = np.mean(x_intersects, axis=1)
        # Calcualate average distance to the focus points of the two peripheral rays
        # The synthetic CECE diagnostic only cares about the z plane so we throw out the toroidal plane
        ECE_launch["dist_focus"][i] = np.linalg.norm(x_orig_3D - np.array([x_intersects[1], y_intersects[1], z_intersects[1]]))
    return ECE_launch
        

def line(x0, vec, s):
    return x0 + vec*s


def plot1DECE(wgIn = 8, antenna = 'CECE', dtoECE = 55, 
              freq = 110, project = 'poloidal', corr=1.0, doPlot = True, verb = True):
    """Ray-traces EM beam from (C)ECE waveguides in sector 9 to the plasma
    
    This code plots the paths of central and 1/e^2 intensity rays out of
    the 1D ECE antennas into the plasma. It uses Snell's law in a 2D geometry 
    at the lenses' interfaces 
    
    History:
    199X        I. Classen - original author of "finalSuttrop.m" upon which this code is based
    2017        S. Freethy - translated finalSuttrop into pyhon
    Dec 2019    P. Molina - adds comments + waveguide positions + different projections
                            and absolute machine coordinates
    Jul 2021    P. Molina - adds vacuum window
    Oct 2021    P. Molina - include fix to vacuum window position from J. Friessen
    Oct 12 21   M. Willensdorfer/P. Molina - noticed and fixed bug that prevented rays from entering v window
    
    Parameters
    ---------
    #   wgIn    - gives the waveguide number input into ECE optics
    #             default is 8 for CECE's waveguide position as of Jan 2020
    #             can be anything inside [1.. 12]
    #   antenna - whether one uses the traditional ECE or the CECE antennas
    #   dtoECE  - distance between CECE antenna mouth and 1D ECE w/g mouth in mm
    #             55mm -default - refers to the separation when the RF box is flushed against back of rail (see wiki)
    #   freq    - frequency in GHz
    #   project - poloidal or toroidal plane projection to calc/plot
    #   corr    - a correction to the launching angles to avoid clipping at vaccum window surface. Defaults to 1.0 = no correction
    #   doPlot  - 1 to show plots
    #   verb    - verbose on/off
    
    Returns
    ---------  
    #   x       - radial vector. AUG coordinates
    #   z       - vertical position vector. aug coords.
    #             nx5 matrix defining vertical position of 5 rays: complex ray tracing
    #             2 divergence, 2 waist rays, and one central ray (3rd one)
    #             Find definitions in Harvey et al. Optical Engineering, 54(3), 035105 (2015).
    #   
    """
    
    ###################################
    # Positioning definitions 
    ##################################
    
    # Use a cartesian coordinate system centred at the machine centre.
    # The x axis points in the direction of the axis of the vacuum window of sec9.
    # x = radial dimentsion. x=0 machine centdaccum window sec 9 
    # y negative means counterclockwise (or RHS) around the machine. 
    # z = vertical dimenstion in torus hall. z=0 machine zero. 
    # all dimensions in mm
    
    # Definitions will follow this order:
    
    # wgs|
    # wgs|----(lens1)-----BS-----(lens2)---(lens3)-----V.Win-----plasma --
    # wgs| 
       
    # ECE waveguide/antenna coordinates
    x0      = -5785 # mm +/-4mm
    
    if project == 'poloidal':
        # in this projection the position of the lenses is raised by 75mm
        z0 = 75 # vertical displacement of the 'middle' ECE wgs. wrt machine centre
    elif project == 'toroidal':
        # in this projection, lenses/waveguides are not vertically displaced.
        # important for plotting later on
        z0 = 0 
        #y0 = 0 # by definition
        
    
    # From J. Freissens' measurements Feb 2020. 5780mm to the polarizing grid.
    # Assumes there's a 5mm distance between the waveguides and the grid.
    # N.B.: x0 = 5663 as shown in 1993 W. Suttrop's sketch is hence believed incorrect
    
    # Define antenna positions: 
    # y and z dimensions depend on which waveguide you are using
    # see sketch in CECE wiki from Johannes.
    # Horizontal distance between axis of waveguides 42mm. 27mm outer diameter of waveguides, hence:
    gh      = 15  # horizontal space between waveguides
    # vertical distance between axis of waveguides 37mm. 27mm outer diameter of waveguides, hence:
    gv      = 10  # vertical space between waveguides
    wgRad   = 27.0/2.0  # ECE waveguide outter radius 
    xAn     = x0;       # all waveguides have the same radial position
    # bottom row 
    if wgIn == 8:
        yAn = -(0.5*gh + wgRad)
        zAn = -(2*wgRad + gv) + z0       
    elif wgIn == 6:
        yAn = -((1.5*gh) + (3.0*wgRad))
        zAn = -(2*wgRad + gv) + z0       
    elif wgIn == 1:
        yAn = +(0.5*gh + wgRad)
        zAn = -(2*wgRad + gv) + z0       
    elif wgIn == 2:
        yAn = +((1.5*gh) + (3.0*wgRad))
        zAn = -(2*wgRad + gv) + z0       
    # centre row
    elif wgIn == 7:
        yAn = -(0.5*gh + wgRad)
        zAn = 0 + z0
    elif wgIn == 5:
        yAn = -((1.5*gh) + (3.0*wgRad))
        zAn = 0 + z0
    elif wgIn == 3:
        yAn = -(0.5*gh + wgRad)
        zAn = 0 + z0
    elif wgIn == 4:
        yAn = +((1.5*gh) + (3.0*wgRad))
        zAn = 0 + z0
    # top row
    elif wgIn == 10:
        yAn = -(0.5*gh + wgRad) 
        zAn = (2*wgRad + gv) + z0 
    elif wgIn == 9:
        yAn = -((1.5*gh) + (3.0*wgRad))
        zAn = (2*wgRad + gv) + z0 
    elif wgIn == 11:
        yAn = (0.5*gh + wgRad)
        zAn = (2*wgRad + gv) + z0 
    elif wgIn == 12:
        yAn = +((1.5*gh) + (3.0*wgRad))
        zAn = (2*wgRad + gv) + z0 
    else:
        print('* Warning * cant find that wg. Using wg 8 then..')
        yAn = -(0.5*gh + wgRad)
        zAn =   -(2*wgRad + gv) + z0  
    
    if verb == True:
        print('going with waveguide #', wgIn)
        print('Wg coords: y=', yAn, 'z =', zAn, 'x = ', xAn)
        
    # Define Lens 1 coords
    # wgs|
    # wgs|----(lens1)-----BS-----(lens2)---(lens3)-----V.Win----plasma--
    # wgs| 

    # lens refractive index. HDPE = high density propylene 
    nhdpe   = 1.52   # refractive index at F-band freqs? Source? 
    
    # Lens 1 
    thick1  = 31.2      
    # distance between surface vertices: where the lens surface crosses the optical axis.
    xL1     = -5382     # 403 from the waveguide's 
    # yL1     = 0       # hopefully centered with the v window.
    zL1     = 0 + z0
    RL1     = 2000      # spherical front
    CL1     = [xL1+RL1-thick1/2,zL1]
    RL1b    = 2000          # spherical back
    CL1b    = [xL1-RL1b+thick1/2,zL1]
    nL1     = nhdpe
    # since RL1 >> thick1, these can be considered thin lenses
    
    # Define beam splitter coords
    # currently not supported because it should not affect 1D ECE optics (hopefully)
    
    # Lens 2 
    thick2  = 60.82
    xL2     = -4188      # 1194 from lens 1 above.
    zL2     = 0 + z0
    #spherical front
    RL2     = 1500       # radius of curvature
    CL2     = [xL2+RL2-thick2/2,zL2] #[x,y] position of center of curvature
    #spherical back
    RL2b    = 1500 
    CL2b    = [xL2-RL2b+thick2/2,zL2]
    nL2     = nhdpe

    # Lens 3 
    thick3  = 51.87
    xL3     = -3889      # 299 from lens 2 above      
    zL3     = 0 + z0 
    #spherical front
    RL3     = 2800
    CL3     = [xL3+RL3-thick3/2,zL3]
    #spherical back
    RL3b    = 2800 
    CL3b    = [xL3-RL3b+thick3/2,zL3]
    nL3     = nhdpe
    
    # Define the optically-relevant frames of the surrounding structure    
    #                               outwardStruct   ->|________|
    # wgs|                                              |
    # wgs|----(lens1)-----BS-----(lens2)---(lens3)-----V.Win----plasma--
    # wgs|                                              | 
    #                                                 |--------|
    
    # Define vacuum window    
    nvw          = 1.949 # from Heraus see 1D ECE optics in the CECE wiki.
    windWbot     = 33.0 #[mm] bottom width of vacuum window
    windWtop     = 40.0 #[mm] top width of vacuum window
    xWindInside  = -3577.0 #[mm] distance from machine center to inside side of vacuum window. Straight side.
    # (from plasma side)  from email from Johannes Freisen 2021: 
    # dem Plasma ist die gerade Seite des Vakuum-Fensters zugeneigt und die 
    # Verschraubung ist von außen. Die 3577mm sind der kürzeste Abstand von der 
    # Z-Achse bis zur geraden Seite Vakuum-Fenster Innenseite/Plasmaseite
    xWindOutside = xWindInside - ((windWbot+windWtop)/2.0) # from measurements of AUG 3D CAD drawings. 
    # x position of window facing the outside : i.e. not the lasma. This surface is tilted.
    # See Johannes. Believed to be +/-5mm accurate
    
    # yWindow     = 0       # by definition above
    # zWindow     = 0 + z0   # by definition of sect9 frame of reference
    diamWin     = 185*2 # window and structure diameters same to within 1mm    
    # define slope of tilted section, useful for ray tracing later
    # define y=mx+b coordinates of tilted window side. obtained by arithmetics on window dimensions
    m, b = (-diamWin/(windWtop-windWbot)), (+(z0+diamWin/2.0)-(-diamWin/(windWtop-windWbot))*(xWindInside-windWtop))
    # m is slope from tilted window, b using the window's top right corner point    
    
    xOutStruct  = -3617  # 'outward' structure meaning outside from the plasma perspective  
    xInStruct   = -3430    

    xEnd       = -946 # HFS reactor wall radius. From diaggeom
    
    ################################################
    #   Start the ray-tracing:
    ################################################

    c   = const.c
    f   = freq*1e9              # frequency: min 86e9 max 140e9;
    wl  = (c/f)*1000            # wavelength in mm
    
    # define beam out of the antenna
    if antenna == 'ECE':
        w0  = 8.3
        xf  = 0 # distance to focal point. Assume 0.
        # effective beam waist at the antennae in mm, derived from antenna pattern measurements
    elif antenna == 'CECE':
        print('Using CECE antenna...')
        if project == 'poloidal':
            # Feb 2020. Have a rectangular D band horn.
            wMouth  = 11*0.5  # this is the D-band rect horn width at the antenna mouth
            # from measurements of D-band smooth-walled rectangular horn. E-plane dimension 11mm.  
            # Goldsmith page 169. Table 7.1 w/b = 0.5 rect horn. b = 11mm. wo = 11*0.5
            flareAng    = np.arctan(((11/2)-(0.8255/2))/44) # 0.8255 is the height of the waveguide in D -band for that antenna
            l1          = np.sqrt((( (11/2)-(0.8255/2) )**2)+(44**2) ) # main horn slant
            l2          = (0.8255/2)/np.sin(flareAng)
            RcAtMouth   = l1+l2
            # from Goldsmitdh pg 167 beam focal radius and distancing z from focus.
            w0 = wMouth/np.sqrt(1+ ( ((np.pi*(wMouth**2))/(wl*RcAtMouth))**2 ))
            xf = RcAtMouth/( 1+ ( (wl*RcAtMouth)/(np.pi*(wMouth**2)))**2) 
            # because of this distance from focal point
            # dtoECE should come from config file of CECE diag
            # depends on the box position. standard 55 is flush against back of rail
            xAn = xAn - dtoECE - xf  # minus moves them away from tokamak
            print('Antenna radial position ', xAn, '[mm]')
        elif project == 'toroidal':
            # July 2021. Have a rectangular D band horn. Compute the beam radius in the toroidal dimension
            print('**Warning**: chosen toroidal antenna pattern. \n\r **Careful** vacuum window not implemented well! ')
            wMouth  = 15*0.35  # pg 163 rect horns. Goldsmidth chp 7.
            # 15 from measurements of inside of rectangular horn
            # Goldsmith page 163. Table 7.1 w/a = 0.35 rect horn. a = 15mm. wx = a*0.35
            flareAng    = np.arctan(((15/2)-(1.65/2))/44) # 1.65mm is the width of the waveguide in D -band for that antenna
            #44mm is the horizontal length of the antenna
            l1          = np.sqrt((( (15/2)-(1.65/2) )**2)+(44**2) ) # main horn slant
            l2          = (1.65/2)/np.sin(flareAng)
            RcAtMouth   = l1+l2
            # from Goldsmitdh pg 167 beam focal radius and distancing z from focus.
            w0 = wMouth/np.sqrt(1+ ( ((np.pi*(wMouth**2))/(wl*RcAtMouth))**2 ))
            xf = RcAtMouth/( 1+ ( (wl*RcAtMouth)/(np.pi*(wMouth**2)))**2) 
            # because of this distance from focal point
            # dtoECE should come from config file of CECE diag
            # depends on the box position. standard 55 is flush against back of rail
            xAn = xAn - dtoECE - xf  # minus moves them away from tokamak
            print('Antenna radial position ', xAn, '[mm]')
            
    if verb==True:
        print('beam radius at onset: ', w0, '[mm]')
        print('distance to beam waist: ', xf, '[mm]')

    lR  = (np.pi*(w0**2))/(wl)  # rayleigh length
    if verb == True:
        print('Rayleigh length: ', lR, '[mm]')
         
    alfa0 = np.arctan(wl/(np.pi*w0))
    # see https://www.rp-photonics.com/gaussian_beams.html
    # theta = lambda/(pi*wo) seen here assumes tan(theta) ~= theta at small angles
        
    if verb == True:
        print('divAngle: ',np.degrees(alfa0), '[deg]')
    
    # Define number of rays and starting angles for each:
    #initial angles. Plot the 1/e^2 intensity by plotting +/-alfa0 angles
    alfa    = [-alfa0*1.0, 0.0, 0.0, 0.0, alfa0*1.0]
    # these are [-divergence ray, -waist ray, chief ray, + waist ray, + divergence ray]
    
    # ** correction for vignetting ** guess angle that is cleared by structure
    #corr    = 0.83 # now accept as input
    alfa    = np.array(alfa)*corr
    # ** correction for vignetting **
    
    nbeams = len(alfa)
    print('nbeams: ', str(nbeams))
    dx      = 0.1 # step size [mm]
    
    if project == 'poloidal':
        x = [xAn]*np.ones(nbeams)    # defined from wgIn in plot1DECE
        z = [zAn, zAn-w0, zAn, zAn+w0, zAn ]
    elif project == 'toroidal':
        x = [xAn]*np.ones(nbeams)   
        z = [yAn, yAn-w0, yAn, yAn+w0, yAn]

    inlens1 = np.zeros(nbeams) #flag: 0=beam is outside lens 1=beam is inside lens
    inlens2 = np.zeros(nbeams)
    inlens3 = np.zeros(nbeams)
    invw    = np.zeros(nbeams)

    nx      = int(abs(xEnd-xAn)/dx) # number of points to calculate
    xx      = np.zeros(nx)
    xx[0]   = x[0]
    x       = xx
       
    zz      = np.zeros([nx,nbeams])
    zz[0,:] = z
    z       = zz
    
    for i in range(nx-2):
        
        # This loop propagates the beam from the antenna to the AUG HFS wall
        # it moves the radial vector until a lens is encountered or exited.
        # It uses Snell's law at the interface to bend the ray appropriately.
        # ** assumes lens material is homogeneus and has a constant refr.index.
        # ** 'rays' are independant, so infinitely small focal point is unphysical
        
        x[i+1]   = x[i] + dx
        z[i+1,:] = z[i,:] + (dx*np.tan(alfa)) 
        
        for k in range(nbeams):
            
            #if spherical front surface lens 1 is passed
            if (((x[i+1]-CL1[0])**2+(z[i+1,k]-CL1[1])**2) < RL1**2) and (((x[i]-CL1[0])**2+(z[i,k]-CL1[1])**2) > RL1**2): 
                #print('got in 1')
                #print('x=', x[i+1], 'y=', z[i+1,k])
                alfaR      = np.arctan((z[i+1,k]-CL1[1])/(x[i+1]-CL1[0])) + np.pi
                #print(' alfa R: ',str(np.rad2deg(alfaR)))
                alfai      = np.pi - alfaR + alfa[k]
                #print(' alfa in: ',str(np.rad2deg(alfai)))
                alfao      = np.arcsin(np.sin(alfai)/nL1)
                alfa[k]    = alfaR + alfao - np.pi
                inlens1[k] = inlens1[k] + 1
      
            #if spherical back surface lens 1 is passed
            elif inlens1[k] and (((x[i+1]-CL1b[0])**2+(z[i+1,k]-CL1b[1])**2)>RL1b**2):
                #print('got out 1')
                #print('x=', x[i+1], 'y=', z[i+1,k])
                alfaR      = np.arctan((z[i+1,k]-CL1b[1])/(x[i+1]-CL1b[0]))
                #print(' alfa R: ',str(np.rad2deg(alfaR)))
                alfai      = alfaR - alfa[k] 
                alfao      = np.arcsin(nL1*np.sin(alfai))
                alfa[k]    = alfaR - alfao
                inlens1[k] = 0
            
            #if spherical front surface lens 2 is passed
            elif (((x[i+1]-CL2[0])**2+(z[i+1,k]-CL2[1])**2) < RL2**2) and (((x[i]-CL2[0])**2+(z[i,k]-CL2[1])**2) > RL2**2):
                alfaR      = np.arctan((z[i+1,k]-CL2[1])/(x[i+1]-CL2[0]))+np.pi
                alfai      = np.pi - alfaR + alfa[k]
                alfao      = np.arcsin(np.sin(alfai)/nL2)
                alfa[k]    = alfaR + alfao - np.pi
                inlens2[k] = inlens2[k] + 1
                  
            #if spherical back surface lens 2 is passed
            elif inlens2[k] and (((x[i+1]-CL2b[0])**2+(z[i+1,k]-CL2b[1])**2) > RL2b**2):
                alfaR      = np.arctan((z[i+1,k]-CL2b[1])/(x[i+1]-CL2b[0]))
                alfai      = alfaR - alfa[k] 
                alfao      = np.arcsin(nL2*np.sin(alfai))
                alfa[k]    = alfaR - alfao
                inlens2[k] = 0
             
            #if spherical front surface lens 3 is passed
            elif (((x[i+1]-CL3[0])**2+(z[i+1,k]-CL3[1])**2) < RL3**2) and (((x[i]-CL3[0])**2+(z[i,k]-CL3[1])**2) > RL3**2):
                alfaR      = np.arctan((z[i+1,k]-CL3[1])/(x[i+1]-CL3[0])) + np.pi
                alfai      = np.pi - alfaR + alfa[k]
                alfao      = np.arcsin(np.sin(alfai)/nL3)
                alfa[k]    = alfaR + alfao - np.pi
                inlens3[k] = inlens3[k] + 1
            
            #if spherical back surface lens 3 is passed
            elif inlens3[k] and (((x[i+1]-CL3b[0])**2 + (z[i+1,k]-CL3b[1])**2) > RL3b**2):   
                alfaR      = np.arctan((z[i+1,k] - CL3b[1])/(x[i+1] - CL3b[0]))
                alfai      = alfaR - alfa[k]
                alfao      = np.arcsin(nL3*np.sin(alfai))
                alfa[k]    = alfaR - alfao
                inlens3[k] = 0
                
            #if vacuum window is encountered. Tilted side. Screws go outside vacuum vessel, hopefully!
            elif (x[i+1] > xWindInside-windWtop) and (x[i+1] < xWindInside-windWbot) and (invw[k] == 0):
                # maybe inside vwindow. check if it's on the surface there.
                if (z[i+1,k] < ((m*(x[i+1]-dx))+b)) and ( z[i+1,k] > ((m*(x[i+1]+dx))+b) ):
                    #print('entered window, ray ', str(k))
                    #print('x=', x[i+1], 'y=', z[i+1,k])
                    winHeight   = diamWin 
                    winWidth    = 40-33
                    alfaR      = np.pi+(((np.pi/2.0) - np.arctan(winHeight/winWidth)))
                    #print(' alfa R: ',str(np.rad2deg(alfaR)))
                    alfai      = np.pi - alfaR + alfa[k]
                    #print(' alfa in: ',str(np.rad2deg(alfai)))
                    alfao      = np.arcsin(np.sin(alfai)/nvw)
                    alfa[k]    = alfaR + alfao - np.pi
                    invw[k]    = invw[k] + 1
                        
            #if vacuum window is exited.
            elif (invw[k]==1) and (x[i+1] > xWindInside-dx) and (x[i+1] < xWindInside+dx):
                #print(' exited RHS vacuum window, ray ', str(k))
                #print('x=', x[i+1], 'y=', z[i+1,k])
                # surface angle - flat side of window: atan(CL[1]/CL[0]=infty) = atan(0)
                alfaR       = 0
                alfai       = alfaR - alfa[k] 
                #print(' alfa in: ',str(np.rad2deg(alfai)))
                alfao       = np.arcsin(nvw*np.sin(alfai))
                alfa[k]     = alfaR - alfao
                invw[k]     = 0
    
    #print('Done Snell 2D')  
                     
    ###############################################
    # store and/or plot the result
    ###############################################
        
    if doPlot == 1:
   
        # plt.figure(figsize = (18,9))
        ax = plt.gca() # get current reference
        
        # plot the rays
        plt.plot(x,z[:,0], '.',color = 'orange', markersize=0.6) # divergence ray
        plt.plot(x,z[:,1], '.',color = 'blue', markersize=0.6)   #    waist ray
        plt.plot(x,z[:,2], '.',color = 'black', markersize=0.6)  #   chief ray
        plt.plot(x,z[:,3], '.',color = 'green', markersize=0.6)   #  waist ray
        plt.plot(x,z[:,4], '.',color = 'red', markersize=0.6) #  divergence ray
        
        
        # plot waveguides
        wgLen   = 58 # length of waveguide bit that enters the ECE box
                
        if project == 'poloidal':
            # plot first, the current waveguide
            rec     = pat.Rectangle((xAn-wgLen,zAn-wgRad), wgLen, 2*wgRad, 
                                fill = True, color ='gold')
            ax.add_patch(rec)
            #plot other waveguides (only frame)            
            # central row waveguides
            rec     = pat.Rectangle((x0-wgLen,z0-wgRad), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec)
            # top waveguides
            rec     = pat.Rectangle((x0-wgLen,z0+wgRad+gv), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec) 
            # bottom waveguides
            rec     = pat.Rectangle((x0-wgLen,z0-(3*wgRad)-gv), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec) 
        elif project == 'toroidal':
            # plot first, the current waveguide
            rec     = pat.Rectangle((xAn-wgLen,yAn-wgRad), wgLen, 2*wgRad, 
                                fill = True, color ='gold')
            ax.add_patch(rec)
            #plot other waveguides (only frame)            
            # left-most
            rec     = pat.Rectangle((x0-wgLen,-(1.5*gh+4*wgRad)), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec)
            # first left
            rec     = pat.Rectangle((x0-wgLen,-(0.5*gh+2*wgRad)), wgLen, 2*wgRad,
                                     fill = False, color ='gold')
            ax.add_patch(rec) 
            # first right
            rec     = pat.Rectangle((x0-wgLen,+(0.5*gh)), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec)
            # right-most
            rec     = pat.Rectangle((x0-wgLen,+(1.5*gh+2*wgRad)), wgLen, 2*wgRad, 
                                fill = False, color ='gold')
            ax.add_patch(rec)
            
        # plot lenses
        #spherical surface lens 1
        anglef = (np.linspace(8,-8)+180)*np.pi/180
        xf = RL1*np.cos(anglef)+CL1[0]
        zf = RL1*np.sin(anglef)+CL1[1]
        plt.plot(xf,zf, '-b')
        #spherical back surface lens 1
        anglef = (np.linspace(8,-8))*np.pi/180
        xf = RL1b*np.cos(anglef)+CL1b[0]
        zf = RL1b*np.sin(anglef)+CL1b[1]
        plt.plot(xf,zf, '-b')
        # spherical surface lens 2
        anglef = (np.linspace(12,-12)+180)*np.pi/180
        xf = RL2*np.cos(anglef)+CL2[0]
        zf = RL2*np.sin(anglef)+CL2[1]
        plt.plot(xf,zf, '-b')
        #spherical back surface lens 2
        anglef = (np.linspace(12,-12))*np.pi/180
        xf = RL2b *np.cos(anglef)+CL2b[0]
        zf = RL2b *np.sin(anglef)+CL2b[1]
        plt.plot(xf,zf, '-b')
        #spherical surface lens 3
        anglef = (np.linspace(8,-8)+180)*np.pi/180
        xf = RL3*np.cos(anglef)+CL3[0]
        zf = RL3*np.sin(anglef)+CL3[1]
        plt.plot(xf,zf, '-b')
        #spherical back surface lens 3
        anglef = (np.linspace(8,-8))*np.pi/180
        xf = RL3b *np.cos(anglef)+CL3b[0]
        zf = RL3b *np.sin(anglef)+CL3b[1]
        plt.plot(xf,zf, '-b')

        # plot vacuum window + structure around
        # vacuum window is a trapezoid 
        xpoints = [xWindInside, xWindInside-windWbot, xWindInside-windWtop, xWindInside] # ccw on the window
        ypoints = [z0-diamWin/2.0,   z0-diamWin/2.0,  z0+diamWin/2.0,         z0+diamWin/2.0]
        ax.add_patch(pat.Polygon(xy=list(zip(xpoints,ypoints)), fill=True, color='cyan'))
        
        # structure blocks
        bh = 100 # block height: arbitrary
        rec = pat.Rectangle((xOutStruct,z0+(diamWin/2)), (xInStruct-xOutStruct),
                             bh, fill = True, color ='blue')
        ax.add_patch(rec) 
        rec = pat.Rectangle((xOutStruct,z0-(diamWin/2)-bh), (xInStruct-xOutStruct),
                             bh, fill = True, color ='blue')
        ax.add_patch(rec) 

        # plot the HFS backwall
        plt.plot((xEnd, xEnd), (+1000,-1000), '-.b')
        plt.text(xEnd+10, 0, 'HFS backwall', fontsize=15)
        
        #saveas(gcf,'1D_ECE_optics.eps','epsc')

        plt.grid(True, which='both')
        plt.axis('equal')
        plt.ylim((-1000,1000))
        plt.xlim([-6000, 0])
        
        if project == 'poloidal':
            plt.title('1D ECE optics 2019 - Poloidal projection - looking at sec 9 from the right. Waveguide: ' + str(wgIn), fontsize = 15)
            plt.xlabel('Radius from machine centre [mm]', fontsize = 15)
            plt.ylabel('Vertical dimension from machine centre [mm]',fontsize = 15)
        elif project == 'toroidal':
            plt.title('1D ECE optics 2019 - Toroidal projection - looking down into tokamak. Waveguide: '+ str(wgIn), fontsize = 15)
            plt.xlabel('Radius from machine centre [mm]', fontsize = 15)
            plt.ylabel('Toroidal displacement from axis of sec 9 vacuum window [mm]', fontsize = 15)
        
        plt.show()
        
    # return paths.
    # Transform into meters. Return the positive radius vector.
   
    x = x/1000
    z = z/1000
    x = np.abs(x)
    
    return x,z
    

def validateQparams(freq=110, dtoECE=55, rRes = 2040, zRes = 78.3):
    # this routine calculates the q parameters of a Gaussian
    # beam in the poloidal plane coming in from an equatorial antenna in order
    # to validate the Snell's law approach taken above
    
    # rRes and zRes are the points in the vessel where the resonance lies in mm
    
    # being with position definitions: see full comments in function above.
    x0      = -5785     # same as above
    xAn     = x0   # all waveguides have the same radial position
#   wgIn    = 7     # equatorial w/g
#   zAn     = 0     # assume this for validation purposes... 
#   z0      = 0   
    # Define Lens 1 coords
    # wgs|
    # wgs|----(lens1)-----BS-----(lens2)---(lens3)-----V.Win----plasma--
    # wgs| 

    # lens refractive index. HDPE = high density propylene 
    nhdpe   = 1.52   # refractive index at F-band freqs? Source?   
   # Lens 1 
    thick1  = 31.2      
    # distance between surface vertices: where the lens surface crosses the optical axis.
    xL1     = -5382     # 403 from the waveguide's 
    # yL1     = 0       # hopefully centered with the v window.
    RL1     = 2000      # spherical front
    nL1     = nhdpe
    # since RL1 >> thick1, these can be considered thin lenses
    
    # Define beam splitter coords
    # currently not supported because it should not affect 1D ECE optics (hopefully)
    
    # Lens 2 
    thick2  = 60.82
    xL2     = -4188      # 1194 from lens 1 above.
    #spherical front
    RL2     = 1500       # radius of curvature
    nL2     = nhdpe

    # Lens 3 
    thick3  = 51.87
    xL3     = -3889      # 299 from lens 2 above      
    #spherical front
    RL3     = 2800
    nL3     = nhdpe
    
    # Define the optically-relevant frames of the surrounding structure    
    #                               outwardStruct   ->|________|
    # wgs|                                              |
    # wgs|----(lens1)-----BS-----(lens2)---(lens3)-----V.Win----plasma--
    # wgs|                                              | 
    #                                                 |--------|
    
    # Define location of the vacuum window.    
    xWindow     = -3577 # from measurements of AUG 3D CAD drawings. 
    # the width will be validated at the window position.
    
    c   = const.c
    f   = freq*1e9              # frequency: min 86e9 max 140e9;
    wl  = (c/f)*1000            # wavelength in mm
    
    # Feb 2020. Have a rectangular D band horn.
    # dtoECE  = 55 # flushed against back of slider. 
    wMouth  = (11)*0.5  # this is the width at the antenna mouth
    
    # from measurements of D-band smooth-walled rectangular horn. E-plane dimension 11mm.  
    # Goldsmith page 169. Table 7.1 w/b = 0.5 rect horn. b = 11mm. wo = 11*0.5
    # horn slant length from Goldsmidth chp 7
    #RcAtMouth     = 47.86127
    flareAng    = np.arctan(((11/2)-(0.8255/2))/44) # 0.8255 is the height of the waveguide in D -band for that antenna
    l1          = np.sqrt((( (11/2)-(0.8255/2) )**2)+(44**2) ) # main horn slant
    l2          = (0.8255/2)/np.sin(flareAng)
    RcAtMouth   = l1+l2
    
    # from Goldsmitdh pg 167 beam focal radius and distancing z from focus.
    w0 = wMouth/np.sqrt(1+ ( ((np.pi*(wMouth**2))/(wl*RcAtMouth))**2 ))
    xf = RcAtMouth/( 1+ (( (wl*RcAtMouth)/(np.pi*(wMouth**2)))**2)) 
    # because of this distance from focal point
    # dtoECE should come from config file of CECE diag
    # depends on the box position. standard 55 is flush against back of rail
    xAn = xAn - dtoECE - xf  # minus moves them away from tokamak
    print('** Antenna radial position ', xAn, '[mm]')
    print('Frequency in: ', freq, '[GHz]')
    print('wavelength: ', wl, '[mm]')
    print('beam radius at onset: ', w0, '[mm]')
    print('distance to beam waist: ', xf, '[mm]')

    lR  = (np.pi*(w0**2))/(wl)  # rayleigh length
    print('Rayleigh length: ', lR, '[mm]')
         
    alfa0 = np.arctan(wl/(np.pi*w0)) 
    print('Initial divAngle: ',np.degrees(alfa0), '[deg]')
    
    # distances between elements
    # distoL1 = np.abs(xL1-xAn) # thin lens
    distoL1 = np.abs(xL1-xAn)-(thick1/2) 
    print('distance to Lens 1 [mm]: ', str(distoL1))
    
    # distoL2 = np.abs(xL2-xL1) # thin lens
    distoL2 = np.abs(xL2-xL1)-(thick1/2)-(thick2/2)
    # playing with wg=8. distances barely change, effect of 0.01 on width :-(
    #distoL2 = 1194.2
    print('distance to Lens 2 [mm]: ', str(distoL2))
    
    # distoL3 = np.abs(xL3-xL2) # thin lens
    distoL3 = np.abs(xL3-xL2)-(thick2/2)-(thick3/2)
    # distoL3 = 299.108
    print('distance to Lens 3 [mm]: ', str(distoL3))
    
    # distoWin = np.abs(xWindow-xL3) # thin lens
    distoWin = np.abs(xWindow-xL3)-(thick3/2)
    #distoWin = 312.145
    print('distance to window [mm]: ', str(distoWin))
    
    # start q parameter calculations
    # q0real = R = infinity
    q0imag  = np.pi*(w0**2)/wl # pg 16 Goldsmidth
    q0      = q0imag*1j 
    qatL1   = q0 + distoL1 # ABCD: A=1, B=L, C=0, D=1
    watL1   = np.sqrt(wl/(np.pi*np.imag(-1/qatL1)))
    print('\n beam rad at L1: ', str(watL1))
    # R   = 1/(np.real(1/qatL1))
    # print(' \n beam rad of curv after L1 ', str(R), '[mm]')   
    # based on pag 43 of Goldsmidth - Quasioptical system. 
    # Thick lens ABCD A=1+(n2-n1)*d/(n2*R1) B=dn1/n2, C= -1/f-correction D=1. Given that R >> d
    f1      = RL1/(2*(nL1-1)) 
    # qaftL1  = (qatL1)/( ((-1/f1)*qatL1) + 1) # thin lens approx
    # print('\n q after L1 thin:  ' + str(qaftL1))
    qaftL1  = ((qatL1*(1+(nL1-1)*thick1/(nL1*RL1)))+(thick1/nL1))/( (((-1/f1)-((thick1*(nL1-1)**2)/(1*nL1*RL1*RL1)))*qatL1) + (1+((1-nL1)*thick1/(nL1*RL1))) )
    # print('\n q after L1 ' + str(qaftL1))
    # move towards lens 2
    qatL2   = qaftL1 + distoL2
    watL2   = np.sqrt(wl/(np.pi*np.imag(-1/qatL2)))
    print('\n beam rad at L2: ', str(watL2))
    f2      = RL2/(2*(nL2-1))
    # qaftL2  = (qatL2)/( ((-1/f2)*qatL2) + 1) # thin lens approx
    qaftL2  = ((qatL2*(1+(nL2-1)*thick2/(nL2*RL2)))+(thick2/nL2))/( (((-1/f2)-((thick2*(nL2-1)**2)/(1*nL2*RL2*RL2)))*qatL2) + (1+((1-nL2)*thick2/(nL2*RL2))) )
    # print('\n q after L2 ' + str(qaftL2))
    # move towards lens 3
    qatL3   = qaftL2 + distoL3
    watL3   = np.sqrt(wl/(np.pi*np.imag(-1/qatL3)))
    print('\n beam rad at L3: ', str(watL3))
    f3      = RL3/(2*(nL3-1))
    # qaftL3  = (qatL3)/( ((-1/f3)*qatL3) + 1) # thin lens approx
    qaftL3  = ((qatL3*(1+(nL3-1)*thick3/(nL3*RL3)))+(thick3/nL3))/( (((-1/f3)-((thick3*(nL3-1)**2)/(1*nL3*RL3*RL3)))*qatL3) + (1+((1-nL3)*thick3/(nL3*RL3))) )
    # print('\n q after L3 ' + str(qaftL3))
    # move towards lens 3
    
    #now, check when it would focus after this
    # here, find for further validation, the location of the focal point
    # from page 35 of Goldsmith, if you have the w and R, you can get wo and z
    w   = np.sqrt(wl/(np.pi*np.imag(-1/qaftL3))) 
    print(' \n beam rad after L3 ', str(w), '[mm]')    
    R   = 1/(np.real(1/qaftL3))
    print(' \n beam rad of curv after L3 ', str(R), '[mm]')    

    wo_focal    = w/np.sqrt(1+((np.pi*(w**2)/(wl*R))**2))
    d_to_focal  = R/(1+(((wl*R)/(np.pi*(w**2)))**2))
    d_absolute  = xL3+thick3/2-d_to_focal
    print('beam radius at focus after L3 [mm]: ', str(wo_focal))
    print('distance to focus after L3 [mm]: ', str(d_to_focal))
    print(' absolute value of focal point in R coords: ', str(d_absolute))
    
    
    # move towards v window
    qatWin  = qaftL3 + distoWin
    w   = np.sqrt(wl/(np.pi*np.imag(-1/qatWin))) 
    print(' \n beam rad at vacuum window ', str(w), '[mm]')    
    R   = 1/(np.real(1/qatWin))
    print(' Rad of curvature at vacuum window ', str(R), '[mm]')    

    # validated to be okish
    # Tues 31 of March. This gave 198.9 vs code drawing using w/g 7
    # gave 194.3. 4mm error, acceptable.
    # don know which is wrong: the snell by starting from a constant angle
    # or the Gaussian for thinking the lenses are thin and without structure
    # March 2021: the small different leads to 10-15cm error on the focal point with the complex ray tracing. So a small difference is problematic here.
    # Currently believing that complex ray tracing is the best way forward
    
    # anyhow, use this to estimate width of Gaussian in the plasma. 
    # during shot 36974, center of channels was at about R=2.040m
    delR        = np.abs(xWindow+rRes)
    delZ        = zRes-78.3 
    # 78.3 by looking at the w/g8 output intersecting window 
        
    disWinRes   = np.sqrt(delR**2+delZ**2)
    
    qatRes      = qatWin + disWinRes
    wAtRes      = np.sqrt(wl/(np.pi*np.imag(-1/qatRes))) 
    print('after moving into the plasma: ', disWinRes, '[mm]')
    print(' beam radius at resonance ', str(wAtRes), '[mm] \n')
    

if(__name__ == "__main__"):
    # plot1DECE(wgIn = 8, antenna = 'CECE', dtoECE = 55, 
    #           freq = 110, project = 'toroidal', doPlot = True, verb = True)
    ECE_launch = get_ECE_launch_v2(8, "CECE", 0.055, np.array([110.4e9, 115.e9]),  np.array([100.e6, 100.e6]))
    print(ECE_launch)
    print(ECE_launch["width"]/2.0)