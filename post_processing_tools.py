"""Set of functions used in the post-processing of results from
the TORBEAM and RELAX codes.
"""

__author__ = 'Omar Maj (omaj@ipp.mpg.de)'
__version__ = '$Revision: $'
__date__ = '$Date: $'
__copyright__ = ' '
__license__ = 'Max-Planck-Institut fuer Plasmaphysik'

# Import statements
import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# customize graphics libraries
params = {'axes.labelsize' : 18,
          'text.fontsize'  : 18,
          'legend.fontsize': 12,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'text.usetex'    : False,
          'lines.linewidth': 1.5}
rcParams.update(params)


# Estimate the temperature anisotropy
def estimate_anisotropy(xgrid, ygrid, F):
    """Given the distribution function F(x,y) function of the normalized
    momentum x and momentum pitch-angle y, this estimate the ratio
    
        xi = 0.5 (E_par / E_perp) - 1
        
    where E_par and E_perp are (proportional to) the parallel and 
    perpendicular energies, hence, xi = 0. for an isotropic distribution.

    USAGE:
    
      xi = estimate_anisotropy(xgrid, ygrid, F)

    INPUT:
    
      - xgrid:
        float ndarray shape = (nptx), normalized momentum grid;
        
      - ygrid:
        float ndarray shape = (npty), momentum pitch-angle grid;

      - F:
        float ndarray shape = (nsurf, nptx, npty), values of the 
        distribution function of the (x,y) grid for every magnetic 
        surface labelles by isurf = 0:nsurf-1.

    OUTPUT:

      - xi:
        float ndarray shape = (nsurf), temperature anisotropy at the
        magnetic surface labelled by isurf = 0:nsurf-1.
    """

    # Extract parameters
    nsurf, nptx, npty = np.shape(F)

    # Integration limits
    x0 = xgrid[0]
    x1 = xgrid[nptx-1]
    y0 = ygrid[0]
    y1 = ygrid[npty-1]

    # Initialize the arrays
    integrand1 = np.empty([nptx, npty])
    integrand2 = np.empty([nptx, npty])
    xi = np.empty([nsurf])    

    # Loop over magnetic surfaces
    for isurf in range(0, nsurf): 
        # ... build the integrands for energies ...
        for ix in range(0, nptx):
            x = xgrid[ix]
            for iy in range(0, npty):
                y = ygrid[iy]
                integrand1[ix, iy] = x**4 * \
                    (np.sin(y))**3 * F[isurf, ix, iy]
                integrand2[ix, iy] = x**4 * \
                    np.sin(y) * np.cos(y)**2 * F[isurf, ix, iy]

        # ... interpolation objects ...
        func1 = RectBivariateSpline(xgrid, ygrid, integrand1)
        func2 = RectBivariateSpline(xgrid, ygrid, integrand2)
        # ... estimate energies ...
        Eperp = func1.integral(x0, x1, y0, y1)
        Epar = func2.integral(x0, x1, y0, y1)
        # estimate the anisotropy index
        xi[isurf] = 0.5 * (Eperp / Epar) - 1.

    # Return the result
    return xi

# Estimate Harvey parameter
def Harvey(relax_results):

    """Compute (dP/dV)*n^-2 where dP/dV is the power deposition in W/cm^3 
    and n is the electron density in 10^13 cm^-3."""

    # Maximum dP/dV converted from kW/m^3 to W/cm^3
    dP_dV = 1.e-3 * max(relax_results.dPec_dV.val)
    
    # Maximum of the density in units 10^13 cm^-3
    n = 1.e-13 * max(relax_results.Ne.val)

    # Return
    return dP_dV / (n * n)

# Current drive efficiency
def CDefficiency(Je, dPdV):
    
    """Given the current density and the power deposition profiles,
    compute the current drive efficiency.

    This returns a marked array.
    """

    # Mark meaningless data
    masked_Je = np.ma.masked_where(abs(dPdV) < 1.e-10, Je)

    # Compute the raw profile
    efficiency = masked_Je / dPdV

    # Return
    return efficiency

# Build the parallel and perpendicular momenta on the equatorial plane
def momenta_on_equatorial_plane(x, y):
    
    """ Comment this!
    """

    # Extract number of grid points
    nptx = np.size(x)
    npty = np.size(y)

    # Build the grid in pll (parallel momentum) and in 
    # pxx (perpendicular momentum)
    pll = np.empty([nptx, npty])
    pxx = np.empty([nptx, npty])
    for ix in range(0, nptx):
        for iy in range(0, npty):
            # normalized parallel momentum
            pll[ix, iy] = x[ix] * np.cos(y[iy])
            # normalized perpendicular momentum
            pxx[ix, iy] = x[ix] * np.sin(y[iy])

    # Exit
    return pll, pxx

# Remapping of the distribution function
def Fe_remapped(x, y, Fe, xi):
    
    """ Comment this!
    """

    # First generate an interpolation object on the rectangular x-y grid
    Fe_int = RectBivariateSpline(x, y, Fe)
    
    # Initialize the arrays for pll, pxx on the point of crossing
    pmax = max(x)
    npts = 100
    pll = np.linspace(-pmax, +pmax, 2 * npts)
    pxx = np.linspace(0., pmax, npts)
    pll, pxx = np.meshgrid(pll, pxx)

    # Corresponding equatorial pitch-angle cosine
    mu = np.sqrt((pll**2 + pxx**2 * (xi - 1.) / xi) / (pll**2 + pxx**2))
    mu = np.copysign(mu, pll)

    # Remapped coordinates on the equatorial plane
    x_eq = np.sqrt(pll**2 + pxx**2)
    y_eq = np.arccos(mu)
    
    # Remapped distribution function
    Fe_rem = Fe_int.ev(x_eq.flatten(), y_eq.flatten())
    Fe_rem = Fe_rem.reshape(np.shape(pll))

    # Exit
    return pll, pxx, Fe_rem

# Process the distribution function
def preprocess_f(relax_results, raytrece, remapping):

    # Extract data 
    nharm = raytrece['nharm']
    enpar = raytrece['beam_Nparallel']
    omcom = raytrece['beam_omcom']
    bbo = raytrece['beam_bbo']

    # Find out the index of the flux surface where the deposition
    # profile attain its maximum
    icross = 0
    isurf = np.argmax(relax_results.dPec_dV.val)
    rho = round(np.sqrt(relax_results.psi.val[isurf]), 3)

    # Extract grid points from RELAX results
    x = relax_results.x.val
    y = relax_results.y.val

    # Trapping factor, i.e., the ratio of the magnetic field strength
    # at the crossing of the surface to the minimum field on that surface
    xi = bbo[isurf, icross]

    # Normalization factor and normalized ditribution
    norm = max(relax_results.Fe.val[isurf,:,:].flatten())
    Fe = relax_results.Fe.val[isurf,:,:] / norm

    # Generate data for the distribution function in cylindrical coordinates
    # depending on the remapping flag
    if remapping:
        pll, pxx, Fe_cylindrical = Fe_remapped(x, y, Fe, xi)
    else:
        pll, pxx = momenta_on_equatorial_plane(x, y)
        Fe_cylindrical = Fe

    # Estimate N_parallel and omega_c / omega at the surface of 
    # maximum deposition
    Nmax = enpar[isurf,0]
    Omax = omcom[isurf,0]

    # Resonance funtions
    res = lambda xll, xpp: \
        np.sqrt(1. + xll**2 + xpp**2) - nharm * Omax - Nmax * xll

    # return
    return pll, pxx, Fe_cylindrical, res, rho, isurf 



#
# --- Plotting funtions ---
#



# Plotting wrapper
def plotting(relax_results, types_of_plot, use_log, raytrece, 
             remapping=True, torbeam_results=None):
    
    """Wrapper for different type of plots."""

    # Select the type of plot
    if 'profiles' in types_of_plot:
        
        # Plot profiles
        fig = figure(1, figsize=(13,7))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95,
                             wspace=0.4, hspace=0.4)
        plot_profiles(relax_results, torbeam_results, raytrece, fig)

    if 'CD efficiency' in types_of_plot:

        # Plot current drive efficiency
        fig = figure(2)
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        plot_CDefficiency(relax_results, torbeam_results, fig)

    if 'F - contours and sections' in types_of_plot:
        
        # preprocess the distribution function
        pll, pxx, Fe, res, rho, isurf = preprocess_f(relax_results,
                                                     raytrece, remapping)

        # Logaritmic plot
        if 'F - contours and sections' in use_log:
            title = r'$\log f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
            distrib = np.log(Fe)
        else:
            title = r'$f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
            distrib = Fe

        # Plot both contours and slices of the distribution function
        fig = figure(3, figsize=(12,4))
        fig.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.9,
                            wspace=0.4, hspace=0.)
        ax = fig.add_subplot(121, aspect='equal')
        plot_f_contours(ax, pll, pxx, distrib, res, title) 
        ax = fig.add_subplot(122)
        plot_f_sections(ax, relax_results, rho, isurf)

    if 'F - contours' in types_of_plot:
        
        # preprocess the distribution function
        pll, pxx, Fe, res, rho, isurf = preprocess_f(relax_results, 
                                                     raytrece, remapping)

        # Logaritmic plot
        if 'F - contours' in use_log:
            title = r'$\log f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
            distrib = np.log(Fe)
        else:
            title = r'$f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
            distrib = Fe

        # Plot both contours and slices of the distribution function
        fig = figure(4, figsize=(9, 4.3))
        subplots_adjust(top=0.95, bottom=0.1, right=0.99)
        ax = fig.add_subplot(111, aspect='equal')
        plot_f_contours(ax, pll, pxx, distrib, res, title) 

    if 'F - sections' in types_of_plot:
        
        # preprocess the distribution function
        pll, pxx, Fe, res, rho, isurf = preprocess_f(relax_results, 
                                                     raytrece, remapping) 

        # Plot both contours and slices of the distribution function
        fig = figure(5)
        ax = fig.add_subplot(111)
        plot_f_sections(ax, relax_results, rho, isurf) 
         
    if 'F - 3d' in types_of_plot:
        
        # preprocess the distribution function
        pll, pxx, Fe, res, rho, isurf = preprocess_f(relax_results, 
                                                     raytrece, remapping)
        # Logaritmic plot
        if 'F - 3d' in use_log:
            distrib = np.log(Fe)
            title = r'$\log f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
        else:
            title = r'$f_\mathrm{e} (p_\parallel, p_\perp)$ at $\rho = $'
            title += str(rho)
            distrib = Fe

        # plot the distribution function
        fig = figure(6)
        ax = fig.gca(projection='3d')        
        plot_f_3d(ax, pll, pxx, distrib, res, title)

    # show plots
    show()

    # Exit
    return

# Plotting radial profiles
def plot_profiles(relax_results, torbeam_results, raytrece, fig):

    """Plot profiles, namely, temperatures, anisotropy parameter,
    driven current density, and power deposition dP/dV as a funtion
    of the square root of the normalized poloidal flux on fig1 and
    the current drive efficiency on fig2.

    USAGE:

       
    INPUT:
    
       
    """
    

    # Abscissa
    rho_relax = np.sqrt(relax_results.psi.val)
    
    # Extract TORBEAM 
    if torbeam_results != None:
        rho_torbeam, dP_dV_torbeam, Je_torbeam = torbeam_results

    # Extract the power deposition profile obtained from extended rays
    dP_dV_rays = raytrece['dP_dV']
    dV = raytrece['dV']
    
    # Check the total integral of the power deposition profile
    nsurf = np.size(dV)
    Ptot_rays = 0.
    for ipsi in range(nsurf):
        Ptot_rays += dP_dV_rays[ipsi] * dV[ipsi]
    print('Total ray power deposition = {} MW'.format(Ptot_rays))

    # Estimate the temperature anisotropy
    anisotropy = estimate_anisotropy(relax_results.x.val,
                                     relax_results.y.val, 
                                     relax_results.Fe.val)

    # Define axis
    gs1 = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])
    ax3 = fig.add_subplot(gs1[2])
    gs1.tight_layout(fig, pad=4.5, h_pad=1.5, w_pad=0.,
                     rect=[None, None, 0.45, None])
    #
    gs2 = gridspec.GridSpec(2, 1)
    ax4 = fig.add_subplot(gs2[0])
    ax5 = fig.add_subplot(gs2[1])
    gs2.tight_layout(fig, pad=3., h_pad=1.5, w_pad=0.,
                     rect=[0.45, None, None, None])

    # Electron temperature and density
    ax1.plot(rho_relax, relax_results.Tini.val, 'b--')
    ax1.plot(rho_relax, relax_results.Te.val, 'r-')
    ax1.set_xlim([min(rho_relax), max(rho_relax)])
    ax1.grid('on')
    ax1.legend(('Intitial profile', 'RELAX profile'), 'lower left')
    ax1.set_ylabel(r'$T_\mathrm{e}$ [$\mathrm{keV}$]')
    ax1.set_title('Electron temperature')
    #
    ax2.plot(rho_relax, 1.e-13 * relax_results.Nini.val, 'b--')
    ax2.plot(rho_relax, 1.e-13 * relax_results.Ne.val, 'r-')
    ax2.set_xlim([min(rho_relax), max(rho_relax)])
    ax2.grid('on')
    ax2.legend(('Intitial profile', 'RELAX profile'), 'lower left')
    ax2.set_ylabel(r'$N_\mathrm{e}$ [$10^{13} \mathrm{cm}^{-3}$]')
    ax2.set_title('Electron number density')

    # Temperature anisotropy
    ax3.plot(rho_relax, anisotropy, 'r-')
    ax3.set_xlim([min(rho_relax), max(rho_relax)])
    ax3.grid('on')
    ax3.set_xlabel(r'$\rho = \sqrt{\bar{\psi}}$')
    ax3.set_ylabel(r'$\mathcal{E}_\perp / (2 \mathcal{E}_\parallel) - 1$')
    ax3.set_title('Temperature anisotropy')
    ax3.yaxis.major.formatter.set_powerlimits((0,0))

    # Adjust axis
    setp(ax1, 'xticklabels', [])
    setp(ax2, 'xticklabels', [])

    # Flux-surface averages electron current density
    # (RELAX profile converted from A/m^2 to MA/m^2)
    if torbeam_results != None:
        ax4.plot(rho_torbeam, Je_torbeam, 'b--')
    ax4.plot(rho_relax, 1.e-6 * relax_results.Je_fsa.val, 'r-')
    ax4.set_xlim([min(rho_relax), max(rho_relax)])
    ax4.grid('on')
    if torbeam_results != None:
        ax4.legend(('TORBEAM profile', 'RELAX profile'), 'upper left')
    ax4.set_ylabel(
        r'$\langle J_{\mathrm{e}} \rangle$ [$\mathrm{MA}/\mathrm{m}^2$]')
    ax4.set_title('Current density (with momentum conservation)')

    # Power deposition profile
    # (RELAX profile is converted from kW/m^3 to MW/m^3)
    if torbeam_results != None:
        ax5.plot(rho_torbeam, dP_dV_torbeam, 'b--')
    ax5.plot(rho_relax, 1.e-3 * relax_results.dPec_dV.val, 'r-')
    ax5.plot(rho_relax, dP_dV_rays, 'k-.')
    ax5.set_xlim([min(rho_relax), max(rho_relax)])
    ax5.grid('on')
    if torbeam_results != None:
        ax5.legend(('TORBEAM profile', 'RELAX profile', 'extended rays'), 
                   'upper left')
    ax5.set_xlabel(r'$\rho = \sqrt{\bar{\psi}}$')
    ax5.set_ylabel(r'$dP/dV$ [$\mathrm{MW}/\mathrm{m}^3$]')
    ax5.set_title('Power deposition profile')

    # Adjust axes
    setp(ax4, 'xticklabels', [])

    # Adjust the grid
    top = min(gs1.top, gs2.top)
    bottom = max(gs1.bottom, gs2.bottom)
    gs1.update(top=top, bottom=bottom)
    gs2.update(top=top, bottom=bottom)

    # Exit
    return

# Plotting current drive efficiency
def plot_CDefficiency(relax_results, torbeam_results, fig):


    # Abscissa
    rho_relax = np.sqrt(relax_results.psi.val)

    # Extract TORBEAM results
    if torbeam_results != None:
        rho_torbeam, dP_dV_torbeam, Je_torbeam = torbeam_results

    # Plotting
    ax = fig.add_subplot(111)

    if torbeam_results != None:
        efficiency_torbeam = CDefficiency(Je_torbeam, dP_dV_torbeam)
        ax.plot(rho_torbeam, efficiency_torbeam, 'b--')

    efficiency_relax = CDefficiency(1.e-6 * relax_results.Je_fsa.val,
                                        1.e-3 * relax_results.dPec_dV.val)
    ax.plot(rho_relax, efficiency_relax, 'r-')
    ax.set_xlim([min(rho_relax), max(rho_relax)])
    ax.grid('on')

    if torbeam_results != None:
        ax.legend(('TORBEAM profile', 'RELAX profile'), 'upper right')

    ax.set_xlabel(r'$\rho = \sqrt{\bar{\psi}}$')
    units = 'in [$\mathrm{Am}/\mathrm{W}$]'
    ax.set_ylabel(r'$\langle J_\mathrm{e} \rangle / (dP/dV)$' + units)
    ax.set_title('Current drive efficiency')

    # Exit
    return

# Plot the isocontours of the distribution as a function of parallel
# and perpendicular momenta defined at the equatorial plane
def plot_f_contours(ax, pll, pxx, Fe, res, title):

    """Contour plot of the electron distribution function and
    its constant-pitch-angle sections as functions of the normalized
    energy.
    """

    # Estimate levels for the distribution funtion
    Fmax = np.max(Fe.flatten())
    Fmin = np.min(Fe.flatten())
    Flevels = np.linspace(Fmin, Fmax, 50)

    # Contour plot
    c1 = ax.contour(pll, pxx, Fe, levels=Flevels, cmap=cm.jet)
    c2 = ax.contour(pll, pxx, res(pll, pxx), 
                     levels=[0.], colors='k')

##    ax.set_xlim(-0.5, +0.3)
##    ax.set_ylim(0., +0.35)

    ax.set_xlabel(r'$p_\parallel / (m_\mathrm{e} c)$')
    ax.set_ylabel(r'$p_\perp / (m_\mathrm{e} c)$')
    ax.set_title(title)
    colorbar(c1)

    # Exit
    return

# Plot sections of the distribution function at constant pitch angle
def plot_f_sections(ax, relax_results, rho, isurf):
        
    # Normalized momentum (one dimensional)
    x = relax_results.x.val

    # Number of points in pitch angle
    npty = np.size(relax_results.y.val)

    # Normalize Fe and extract logarithm
    norm = max(relax_results.Fe.val[isurf,:,:].flatten())
    Fe = relax_results.Fe.val[isurf,:,:] / norm    
    Fe = np.log(Fe)
    
    # Sections of the distribution function
    Eel0 = 511. # electron rest energy in keV
    Eel = Eel0 * (np.sqrt(1. + x**2) - 1.) # electron kinetic energy
    ax.plot(Eel, Fe[:, 0])
    ax.plot(Eel, Fe[:, int(npty / 2)])
    ax.plot(Eel, Fe[:, npty - 1])
    ax.set_xlabel(r'$E$ [keV]')
    ax.set_ylabel(r'$\log f_\mathrm{e} (p, \theta)$')
    ax.legend((r'$\theta = 0$',
               r'$\theta \approx \pi/2$',
               r'$\theta = \pi$'))
    
    # Exit
    return

# Plot sections of the distribution function at constant pitch angle
def plot_f_3d(ax, pll, pxx, Fe, res, title):
    
    surf = ax.plot_surface(pxx, pll, Fe,
                           rstride=2,
                           cstride=2,
                           cmap=cm.coolwarm,
                           linewidth=0., 
                           shade=False,
                           antialiased=True)
    xlabel(r'$p_\perp /mc$')
    ylabel(r'$p_\parallel /mc$')
    colorbar(surf)
    ax.set_title(title)

    # Exit
    return

#
# --- end of file


