import numpy as np                # trig functs
import scipy.constants as const   # to get physical constants
import matplotlib.pyplot as plt   
import matplotlib.patches as pat  # to draw rectangles
from scipy.interpolate import InterpolatedUnivariateSpline

def plot1DECE(wgIn = 8, antenna = 'CECE', dtoECE = 55, 
              freq = 116, project = 'poloidal', doPlot = True, verb = True):

    # This code plots the paths of central and 1/e^2 intensity rays out of
    # the 1D ECE antennas into the plasma. It uses Snell's law in a 2D geometry 
    # at the lenses' interfaces 
    #plot
    # 199X      I. Classen - original author of "finalSuttrop.m" upon which this code is based
    # 2017      S. Freethy - translated finalSuttrop into pyhon
    # Dec 2019  P. Molina  - adds comments + waveguide positions + different projections
    #                        and absolute machine coordinates
    
    # Inputs
    # --------
    #   wgIn    - gives the waveguide number input into ECE optics
    #             default is 8 for CECE's waveguide position as of Jan 2020
    #             can be anything inside [1.. 12]
    #   antenna - whether one uses the traditional ECE or the CECE antennas
    #   dtoECE  - distance between CECE antenna mouth and 1D ECE w/g mouth in mm
    #             55mm -default - refers to the separation when the RF box is flushed against back of rail (see wiki)
    #   freq    - frequency in GHz
    #   project - poloidal or toroidal plane projection to calc/plot
    #   doPlot  - 1 to show plots
    #   verb    - verbose on/off
    
    # Outputs
    # ---------  
    #   x       - radial vector = 1D
    #   z       - nx3 matrix defining vertical position of 3 rays
    #             bottom, central, and top 1/e^2 intensity Gaussian beam rays
    #   
    
    ###################################
    # Positioning definitions 
    ##################################
    
    # Use a cartesian coordinate system centred at the machine centre.
    # The x axis points in the direction of the axis of the vacuum window of sec9.
    # x = radial dimentsion. x=0 machine centre
    # y = Toroidal dimension. y=0 center of vaccum window sec 9 
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
    
    # Define location of the vacuum window.    
    xWindow     = -3577 # from measurements of AUG 3D CAD drawings. 
    # See Johannes. Believed to be +/-1mm accurate
    # yWindow     = 0       # by definition above
    # zWindow     = 0 +z0   # trivial
    diamWin     = 185*2 # window and structure diameters same to within 1mm
    
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
        # Feb 2020. Have a rectangular D band horn.
        wMouth  = 11*0.5  # this is the D-band rect horn width at the antenna mouth
        # from measurements of D-band smooth-walled rectangular horn. E-plane dimension 11mm.  
        # Goldsmith page 169. Table 7.1 w/b = 0.5 rect horn. b = 11mm. wo = 11*0.5
        RcAtMouth     = 47.86127 # this is the D-band horn radius of curvature at antenna mouth
        # from Goldsmitdh pg 167 beam focal radius and distancing z from focus.
        w0 = wMouth/np.sqrt(1+ ( ((np.pi*(wMouth**2))/(wl*RcAtMouth))**2 ))
        xf = RcAtMouth/( 1+ ( (wl*RcAtMouth)/(np.pi*(wMouth**2)))**2) 
        # because of this distance from focal point
        # dtoECE should come from config file of CECE diag
        # depends on the box position. standard 55 is flush against back of rail
        xAn = xAn - dtoECE - xf  # minus moves them away from tokamak
        print('** new antenna radial position ', xAn, '[mm]')
        
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
    alfa    = [-alfa0, 0, alfa0]
    nbeams = len(alfa)
    dx      = 0.1 # step size [mm]
    
    if project == 'poloidal':
        x = [xAn]*np.ones(nbeams)    # defined from wgIn in plot1DECE
        z = [zAn]*np.ones(nbeams) 
    elif project == 'toroidal':
        x = [xAn]*np.ones(nbeams)   
        z = [yAn]*np.ones(nbeams)

    inlens1 = np.zeros(nbeams) #flag: 0=beam is outside lens 1=beam is inside lens
    inlens2 = np.zeros(nbeams)
    inlens3 = np.zeros(nbeams)

    nx      = int(abs(xEnd-xAn)/dx) # number of points to calculate
    xx      = np.zeros(nx)
    xx[0]   = x[0]
    x       = xx
       
    zz      = np.zeros([nx,nbeams])
    zz[0,:] = z[0]
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
                alfai      = np.pi - alfaR + alfa[k]
                alfao      = np.arcsin(np.sin(alfai)/nL1)
                alfa[k]    = alfaR + alfao - np.pi
                inlens1[k] = inlens1[k] + 1
      
            #if spherical back surface lens 1 is passed
            elif inlens1[k] and (((x[i+1]-CL1b[0])**2+(z[i+1,k]-CL1b[1])**2)>RL1b**2):
                #print('got out 1')
                #print('x=', x[i+1], 'y=', z[i+1,k])
                alfaR      = np.arctan((z[i+1,k]-CL1b[1])/(x[i+1]-CL1b[0]))
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
        if(inlens3[k] == 0 and x[i] < CL3b[0] - RL3b):
            dx_remain = np.zeros(x[i+1:].shape)
            dx_remain[:] = dx
            dx_remain = np.cumsum(dx_remain)
            x[i+1:]   = x[i] + dx_remain
            for k in range(nbeams):
                z[i+1:,k] = z[i,k] + (dx_remain*np.tan(alfa[k]))
            break 
            
 
    #print('Done Snell 2D')  
                     
    ###############################################
    # store and/or plot the result
    ###############################################
        
    if doPlot == 1:
   
        # plt.figure(figsize = (18,9))
        ax = plt.gca() # get current reference
        
        # plot the rays
        for kk in range(nbeams):
            plt.plot(x,z[:,kk], '.',color = 'black', markersize=0.6)
        
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
        rec = pat.Rectangle((xWindow,z0-(diamWin/2)), 1, diamWin, 
                                fill = True, color ='cyan')
        # vwindow thickness set arbitrarily to 1mm. don't really know...
        ax.add_patch(rec) 
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

def getGaussianParamsForChannel(launch, xRef = 3.577):
    launch["R"] = np.zeros(launch["f"].shape)
    launch["phi"] = np.zeros(launch["f"].shape)
    launch["z"] = np.zeros(launch["f"].shape)
    launch["phi_tor"] = np.zeros(launch["f"].shape)
    launch["theta_pol"] = np.zeros(launch["f"].shape)
    launch["dist_focus"] = np.zeros(launch["f"].shape)
    launch["width"] = np.zeros(launch["f"].shape)
    for i, freq in enumerate(launch["f"]):
        # not sure if the two x are identical, treating them separately for now
        xyr, yr = plot1DECE(project="toroidal", freq=freq * 1.e-9, doPlot=False)
        # not sure if the two x are identical, treating them separately for now
        xzr, zr = plot1DECE(project="poloidal", freq=freq * 1.e-9, doPlot=False)
        # Define positions and wave vectors for the central, and one peripheral ray
        yRef = []
        dydxRef = []
        zRef = []
        dzdxRef = []
        for ir in [1,0]:
            # Spline y(x) and z(x) needed for positions at window
            # Use negative x to have the correct directionality
            yxSpl = InterpolatedUnivariateSpline(-xyr, yr[...,ir])
            zxSpl = InterpolatedUnivariateSpline(-xzr,zr[...,ir])
            yRef.append(yxSpl(-xRef))
            zRef.append(zxSpl(-xRef))
            if(ir == 1):
                # torAng = acos((kx, ky).(0,1)) = acos(ky)
                launch["R"][i] = np.sqrt(yRef[-1]**2 + xRef**2)
                launch["phi"][i] = np.rad2deg(np.arctan2(yRef[-1], xRef))
                launch["z"][i] = zRef[-1]
            dydxRef.append(-np.array(yxSpl(-xRef, nu=1)))
            dzdxRef.append(-np.array(zxSpl(-xRef, nu=1)))
            if(ir == 1):
                # Compute arclength for xy plane
                # Need this for the wave vector
                sxy = np.zeros(xyr.shape)
                sxy[1:] = np.sqrt((xyr[1:] - xyr[:-1])**2 + \
                                  (yr[...,ir][1:] - yr[...,ir][:-1])**2)
                sxy = np.cumsum(sxy)
                # Splines for x,y as function of arclength for the wave vector
                xysSpl = InterpolatedUnivariateSpline(sxy, xyr)
                ySpl = InterpolatedUnivariateSpline(sxy,yr[...,ir])
                # Compute ky
                kxy = xysSpl(xRef, nu=1)
                ky = ySpl(xRef, nu=1)
                # Need to normalize the wave vectors to make unit vectors
                ky /= np.sqrt((kxy**2 + ky**2))
                # torAng = 90.e0 - acos((kx, ky).(0,1)) = 90.e0 - acos(ky)
                launch["phi_tor"][i] = 90.e0 - np.rad2deg(np.arccos(ky))
                # Compute arclength for xz plane
                # Need this for the wave vector
                sxz = np.zeros(xzr.shape)
                sxz[1:] = np.sqrt((xzr[1:] - xzr[:-1])**2 + \
                                  (zr[...,ir][1:] - zr[...,ir][:-1])**2)
                sxz = np.cumsum(sxz)
                # Splines for x,z as function of arclength for the wave vector
                xzsSpl = InterpolatedUnivariateSpline(sxz, xzr)
                zSpl = InterpolatedUnivariateSpline(sxz,zr[...,ir])
                # Compute kz
                kxz = xzsSpl(xRef, nu=1)
                kz = zSpl(xRef, nu=1)
                # Need to normalize the wave vectors to make unit vectors
                kz /= np.sqrt((kxz**2 + kz**2))
                # polAng = 90.e0 - acos((kx, kz).(0,1)) = acos(kz)
                launch["theta_pol"][i] = 90.e0 - np.rad2deg(np.arccos(kz))
        # Compute beam width
        launch["width"][i] = np.sqrt((yRef[0] - yRef[1])**2 + \
                                     (zRef[0] - zRef[1])**2)
        print("------- f = {0:3.1f} GHz --------\n".format(freq/1.e9))
        print("Astigmatism: {0:2.1f} cm / {1:2.1f} cm\n".format(1.e2*np.abs(yRef[0] - yRef[1]), \
                                                   1.e2*np.abs(zRef[0] - zRef[1])))
        # Do some linear algebra to find the focus position
        # Set y = y0 + dy/dx (x-xRef) to be equal for peripheral and central ray and solve linear system  of equations
        # Use np.linalg.solve as described in the numpy documentation
        yIntersect = np.linalg.solve(np.array([[dydxRef[0],0],[0,dydxRef[1]]]), np.array(yRef))
        # Note that the x value is already with respect to xRef and not the center of the machine
        yDist = np.sqrt((yIntersect[0])**2 + \
                        (yRef[0] - yIntersect[1])**2)
        zIntersect = np.linalg.solve(np.array([[dydxRef[0],0],[0,dydxRef[1]]]), np.array(yRef))
        # Note that the x value is already with respect to xRef and not the center of the machine
        zDist = np.sqrt((zIntersect[0])**2 + \
                        (zRef[0] - zIntersect[1])**2)
        print(" for f = {0:3.1f} GHz is: \n".format(freq/1.e9))
        print("Distance between z,y focii {0:2.1f} cm \n".format(1.e2*np.abs(yDist-zDist)))
        launch["dist_focus"][i] = 0.5*(yDist + zDist)
    launch["phi"] += (8.5e0) * 22.5
    return launch
    
    
    
    
def validateQparams(freq=118, dtoECE=15):
    # this routine calculates the q parameters of a Gaussian
    # beam in the poloidal plane coming in from an equatorial antenna in order
    # to validate the Snell's law approach taken above
    
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
 #   thick1  = 31.2      
    # distance between surface vertices: where the lens surface crosses the optical axis.
    xL1     = -5382     # 403 from the waveguide's 
    RL1     = 2000      # spherical front
  #  RL1b    = 2000          # spherical back
    nL1     = nhdpe
    # Lens 2 
    xL2     = -4188      # 1194 from lens 1 above.
    RL2     = 1500       # radius of curvature
    nL2     = nhdpe
    # Lens 3 
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
    wMouth  = 11*0.5  # this is the width at the antenna mouth
    # from measurements of D-band smooth-walled rectangular horn. E-plane dimension 11mm.  
    # Goldsmith page 169. Table 7.1 w/b = 0.5 rect horn. b = 11mm. wo = 11*0.5
    RcAtMouth     = 47.86127
    # from Goldsmitdh pg 167 beam focal radius and distancing z from focus.
    w0 = wMouth/np.sqrt(1+ ( ((np.pi*(wMouth**2))/(wl*RcAtMouth))**2 ))
    xf = RcAtMouth/( 1+ ( (wl*RcAtMouth)/(np.pi*(wMouth**2)))**2) 
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
    distoL1 = np.abs(xL1-xAn)
    distoL2 = np.abs(xL2-xL1)
    distoL3 = np.abs(xL3-xL2)
    distoWin = np.abs(xWindow-xL3)
    
    # start q parameter calculations
    # q0real = R = infinity
    q0imag  = np.pi*(w0**2)/wl
    q0      = q0imag*1j 
    # based on pag 43 of Goldsmidth - Quasioptical system. 
    qatL1   = q0 + distoL1 # ABCD: A=1, B=L, C=0, D=1
    # Thin lens ABCD A=1 B=0, C= -1/f D=1
    f1      = RL1/(2*(nL1-1))
    qaftL1  = qatL1*1/((-1/f1)*qatL1 + 1)
    # move towards lens 2
    qatL2   = qaftL1 + distoL2
    f2      = RL2/(2*(nL2-1))
    qaftL2  = qatL2*1/((-1/f2)*qatL2 + 1)
    # move towards lens 3
    qatL3   = qaftL2 + distoL3
    f3      = RL3/(2*(nL3-1))
    qaftL3  = qatL3*1/((-1/f3)*qatL3 + 1)
    
    # move towards v window
    qatWin  = qaftL3 + distoWin
    
    #R   = 1/(np.real(1/qatWin))
    w   = np.sqrt(wl/(np.pi*np.imag(-1/qatWin))) 
    
    print(' width at vacuum window ', str(w), '[mm]')    
    
    # validated to be okish
    # Tues 31 of March. This gave 198.9 vs code drawing using w/g 7
    # gave 194.3. 4mm error, acceptable.
    # don know which is wrong: the snell by starting from a constant angle
    # or the Gaussian for thinking the lenses are thin and without structure
    
    # anyhow, use this to estimate width of Gaussian in the plasma. 
    # during shot 36974, center of channels was at about R=2.040m
    delR        = np.abs(xWindow+2002)
    delZ        = 126.5-78.3 # by looking at the w/g8 output intersecting window 
        
    disWinRes   = np.sqrt(delR**2+delZ**2)
    
    qatRes      = qatWin + disWinRes
    wAtRes      = np.sqrt(wl/(np.pi*np.imag(-1/qatRes))) 
    print('after moving into the plasma: ', disWinRes, '[mm]')
    print(' width at resonance ', str(wAtRes), '[mm]')
    
if(__name__ == "__main__"):
    launch = {"f":np.array([118.e9])}
    launch = getGaussianParamsForChannel(launch)
    print(launch)



    