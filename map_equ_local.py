import sys
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import UnivariateSpline, interp1d, \
                              InterpolatedUnivariateSpline, LinearNDInterpolator, \
                              RectBivariateSpline
import dd_20180130
sf = dd_20180130.shotfile()


class equ_map:


    def __init__(self, shot=None, diag=None, exp='AUGD', ed=0):

        """ Basic constructor. If a diagnostic is specified the shotfile is opened.
        Example: eqm = map_equ.shotfile(29761, 'EQH') """
        self.eq_open = False
        if shot is not None and diag is not None:
            self.Open(shot, diag, exp, ed)


    def __del__(self):

        """ Basic destructor closing the shotfile. """
        self.Close()


    def Open(self, shot, diag='EQH', exp='AUGD', ed=0):

        """
        Input
        ---------
        shot: int
            shot number
        diag: str
            diagnsotics used for mapping (EQI, EQH, ...)
        exp: str
            experiment (AUGD)
        ed:  int
            edition
        """

        self.Close()
        self.sf = sf
        if self.sf.Open(diag, shot, experiment=exp, edition=ed):
# Time grid of equilibrium shotfile
            self.t_eq = self.sf.GetSignal('time')
            nt = len(self.t_eq)
# R, z of PFM cartesian grid
            self.nR = int(self.sf.GetParameter('PARMV', 'M')) + 1
            self.nZ = int(self.sf.GetParameter('PARMV', 'N')) + 1
            self.Rmesh = self.sf.GetSignal('Ri')[: self.nR, 0]
            self.Zmesh = self.sf.GetSignal('Zj')[: self.nZ, 0]
            Lpf = self.sf.GetSignal('Lpf')
            self.lpfp = Lpf%10000 # inside sep
            self.lpfe = (Lpf - self.lpfp)//10000 #outside sep
            self.ind_sort = []
            for jt in range(nt):
                ind_core = np.arange(self.lpfp[jt], -1, -1)
                ind_sol  = np.arange(self.lpfp[jt] + 3, self.lpfp[jt] + 3 + self.lpfe[jt])
                self.ind_sort.append(np.append(ind_core, ind_sol))
            self.eq_open = True
            self.ed = self.sf.edition
        else:
            print(('Problems opening %s:%s:%d for shot %d' %(exp, diag, ed, shot)))

        return self.eq_open


    def Close(self):

        """
        Deleting some attributes, closing dd.shotfile()
        """

        if hasattr(self, 'sf'):
            self.sf.Close()
            del self.sf
        if hasattr(self, 'pfm'):
            delattr(self, 'pfm')
        self.eq_open = False


    def read_pfm(self):

        """
        Reads PFM matrix, only if attribute 'pfm' is missing.
        equ_map.pfm is deleted at any equ_map.Open, equ_map.Close call.
        """

        if not self.eq_open:
            return
        if hasattr(self, 'pfm'):
            return

        nt = len(self.t_eq)
        self.pfm = self.sf.GetSignal('PFM')[: self.nR, : self.nZ, :nt]


    def read_ssq(self):

        """
        Creating dictionary map_equ.equ_map().ssq, containing all 
        SSQ parameters labelled with the corresponding SSQNAM.
        Beware: in different shotfiles, for a given j0 
        SSQ[j0, time] can represent a different variable 
        """

        if not self.eq_open:
            return

        ssqs   = self.sf.GetSignal('SSQ')
        ssqnam = self.sf.GetSignal('SSQnam')

        nt = len(self.t_eq)
        self.ssq = {}
        for jssq in range(ssqs.shape[0]):
            tmp = b''.join(ssqnam[:, jssq]).strip()
            lbl = tmp.decode('utf8') if isinstance(tmp, bytes) else tmp
            if lbl.strip() != '':
                self.ssq[lbl] = ssqs[jssq, :nt]
   

    def read_scalars(self):

        """
        Reads R, z, psi at magnetic axis and separatrix, only if attribute 'r0' is missing.
        equ_map.r0 is deleted at any equ_map.Open, equ_map.Close call.
        """

        if not self.eq_open:
            return

        self.read_ssq()

        nt = len(self.t_eq)
# Position of mag axis, separatrixx

        PFxx = self.sf.GetSignal('PFxx')[:, :nt]
        ikCAT = np.argmin(abs(PFxx[1:, :] - PFxx[0, :]), axis=0) + 1
        if all(PFxx[2]) == 0: ikCAT[:] = 1  #troubles with TRE equilibrium 

        self.orientation = np.sign((PFxx[1:] - PFxx[0]).mean())  #current orientation

# Pol. flux at mag axis, separatrix
        self.psi0 = PFxx[0, :nt]
        self.psix = PFxx[ikCAT, np.arange(nt)]
        self.ipipsi = self.sf.GetSignal('IpiPSI')[:nt]
        

    def get_profile(self, var_name):

        """
        var_name: str
            name of the quantity, like  
            Qpsi       q_value vs PFL 
            Bave       <B>vac 
            B2ave      <B^2>vac
            FFP        ff' 
            Rinv       <1/R>     
            R2inv      <1/R^2>    
            FTRA       fraction of the trapped particles
        """

        if not self.eq_open:
            return

        nt = len(self.t_eq)

        profs = ('TFLx', 'PFL', 'FFP', 'Qpsi', 'Rinv', 'R2inv', 'Bave', 'B2ave')

        if var_name not in profs:
            print('SignalGroup %s unknown' %var_name)
            return None

        var = self.sf.GetSignal(var_name)[:, :nt]

        nrho = np.max(self.lpfp + 1 + self.lpfe)
        if var_name in profs:
            var_sorted = np.zeros((nt, nrho))
            for jt in range(nt):
                sort_wh = self.ind_sort[jt]
                nr = len(sort_wh)
                var_sorted [jt, :nr] = var[sort_wh, jt]
                if nr < nrho:
                    var_sorted[jt, nr:] = var_sorted[jt, nr-1]

            return var_sorted


    def get_mixed(self, var_name):

        """
        var_name: str
            name of the quantity, like  
            Jpol       poloidal current,
            Pres       pressure  
            Vol        plasma Volume  
            Area       plasma Area  
        """

        if not self.eq_open:
            return None, None

        nt = len(self.t_eq)
        mixed = ('Vol', 'Area', 'Pres', 'Jpol')

# Pairs prof, d(prof)/d(psi)

        if var_name not in mixed:
            print('%s not one of the mixed quanties Vol, Area, Pres, Jpol' %var_name)
            return None, None

        tmp = self.sf.GetSignal(var_name)
        var  = tmp[ ::2, :nt]
        dvar = tmp[1::2, :nt]

        nrho = np.max(self.lpfp + 1 + self.lpfe)
        if var_name in mixed:
            var_sorted  = np.zeros((nt, nrho))
            dvar_sorted = np.zeros((nt, nrho))
            for jt in range(nt):
                sort_wh = self.ind_sort[jt]
                nr = len(sort_wh)
                var_sorted [jt, :nr] = var [sort_wh, jt]
                dvar_sorted[jt, :nr] = -dvar[sort_wh, jt]
                if nr < nrho:
                    var_sorted [jt, nr:] = var_sorted[jt, nr-1]

            return var_sorted, dvar_sorted


    def _get_nearest_index(self, tarr):

        tarr = np.minimum(np.maximum(tarr, self.t_eq[0]), self.t_eq[-1])
        idx = interp1d(self.t_eq, np.arange(len(self.t_eq)), kind='nearest')(tarr)
        idx = np.atleast_1d(np.int_(idx))
        unique_idx = np.unique(idx)

        return unique_idx, idx


    def rho2rho(self, rho_in, t_in=None, \
               coord_in='rho_pol', coord_out='rho_tor', extrapolate=False):

        """Mapping from/to rho_pol, rho_tor, r_V, rho_V, Psi, r_a
        r_V is the STRAHL-like radial coordinate

        Input
        ----------
        t_in : float or 1darray
            time
        rho_in : float, ndarray
            radial coordinates, 1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            input coordinate label
        coord_out: str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            output coordinate label
        extrapolate: bool
            extrapolate rho_tor, r_V outside the separatrix

        Output
        -------
        rho : 2d+ array (nt, nr, ...)
        converted radial coordinate
        """

        if not self.eq_open:
            print('Equilibrium not open')
            return

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        rho = np.atleast_1d(rho_in)

        nt_in = np.size(tarr)

        if rho.ndim == 1:
            rho = np.tile(rho, (nt_in, 1))

# Trivial case
        if coord_out == coord_in: 
            return rho

        self.read_scalars()
        
        unique_idx, idx =  self._get_nearest_index(tarr)

        pf = self.get_profile('PFL')
        tf = self.get_profile('TFLx')
        vol, _ = self.get_mixed('Vol')
        if coord_in in ['rho_pol', 'Psi']:
            label_in = pf
        elif coord_in == 'rho_tor':
            label_in = tf
        elif coord_in in ['rho_V','r_V']:
            label_in = vol
            R0 = self.ssq['Rmag']
        elif coord_in == 'r_a':
            R, _ = self.rhoTheta2rz(pf, [0, np.pi], coord_in='Psi')
            label_in = (R[:, 0] - R[:, 1]).T**2
        else:
            raise Exception('unsupported input coordinate')

        if coord_out in ['rho_pol', 'Psi']:
            label_out = pf
        elif coord_out == 'rho_tor':
            label_out = tf
        elif coord_out in ['rho_V','r_V']:
            label_out = vol
            R0 = self.ssq['Rmag']
        elif coord_out == 'r_a':
            R, _ = self.rhoTheta2rz(pf[unique_idx], [0, np.pi],
                                t_in=self.t_eq[unique_idx],coord_in='Psi')
            label_out = np.zeros_like(pf)
            label_out[unique_idx, :] = (R[:, 0] - R[:, 1])**2
        else:
            raise Exception('unsupported output coordinate')

        PFL  = self.orientation*pf
        PSIX = self.orientation*self.psix
        PSI0 = self.orientation*self.psi0
        rho_output = np.zeros_like(rho)

        for i in unique_idx:
  
# Calculate a normalized input and output flux 
#            sort_wh = self.ind_sort[i][:self.lpfp[i]]    # Up to separatrix   

            sep_out, mag_out = np.interp([PSIX[i], PSI0[i]], PFL[i], label_out[i])
            sep_in , mag_in  = np.interp([PSIX[i], PSI0[i]], PFL[i], label_in [i])

            if (abs(sep_out - mag_out) < 1e-4) or (abs(sep_in - mag_in) < 1e-4): #corrupted timepoint
                continue

# Normalize between 0 and 1
            rho_out = (label_out[i] - mag_out)/(sep_out - mag_out)
            rho_in  = (label_in [i] - mag_in )/(sep_in  - mag_in )

            rho_out[(rho_out > 1) | (rho_out < 0)] = 0  #remove rounding errors
            rho_in[ (rho_in  > 1) | (rho_in  < 0)] = 0

            rho_out = np.r_[np.sqrt(rho_out), 1]
            rho_in  = np.r_[np.sqrt(rho_in ), 1]
            

            ind = (rho_out==0) | (rho_in==0)
            rho_out, rho_in = rho_out[~ind], rho_in[~ind]
            
# Profiles can be noisy!  smooth spline must be used
            sortind = np.unique(rho_in, return_index=True)[1]
            w = np.ones_like(sortind)*rho_in[sortind]
            w = np.r_[w[1]/2, w[1:], 1e3]
            ratio = rho_out[sortind]/rho_in[sortind]
            rho_in = np.r_[0, rho_in[sortind]]
            ratio = np.r_[ratio[0], ratio]

            s = UnivariateSpline(rho_in, ratio, w=w, k=4, s=5e-3,ext=3)  #BUG s = 5e-3 can be sometimes too much, sometimes not enought :( 

            jt = idx == i

            rho_ = np.copy(rho[jt])

            r0_in = 1
            if coord_in == 'r_V' :
                r0_in  = np.sqrt(sep_in/ (2*np.pi**2*R0[i]))
            r0_out = 1
            if coord_out == 'r_V' :
                r0_out = np.sqrt(sep_out/(2*np.pi**2*R0[i]))

            if coord_in == 'Psi' :
                rho_  = np.sqrt(np.maximum(0, (rho_ - self.psi0[i])/(self.psix[i] - self.psi0[i])))

# Evaluate spline

            rho_output[jt] = s(rho_.flatten()/r0_in).reshape(rho_.shape)*rho_*r0_out/r0_in

            if np.any(np.isnan(rho_output[jt])):  # UnivariateSpline failed
                rho_output[jt] = np.interp(rho_/r0_in, rho_in, ratio)*rho_*r0_out/r0_in
                
            if not extrapolate:
                rho_output[jt] = np.minimum(rho_output[jt],r0_out) # rounding errors

            rho_output[jt] = np.maximum(0,rho_output[jt]) # rounding errors

            if coord_out  == 'Psi':
                rho_output[jt]  = rho_output[jt]**2*(self.psix[i] - self.psi0[i]) + self.psi0[i]
        
        return rho_output


    def rz2brzt(self, r_in, z_in, t_in=None, diag=False):

        """calculates Br, Bz, Bt profiles
        Input
        ----------
        r_in : ndarray
            R coordinates 
            1D, size(nr_in) or 2D, size (nt, nr_in)
        z_in : ndarray
            Z coordinates 
            1D, size(nz_in) or 2D, size (nt, nz_in)
        t_in : float or 1darray
            time

        Output
        -------
        interpBr : ndarray
            profile of Br on the grid
        interpBz : ndarray
            profile of Bz on the grid
        interpBt : ndarray
            profile of Bt on the grid
        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq
            
        tarr = np.atleast_1d(t_in)
        r_in = np.atleast_2d(r_in)
        z_in = np.atleast_2d(z_in)

        self.read_pfm()

# Poloidal current 

        nt = np.size(tarr)

        if np.size(r_in, 0)!= nt:
            r_in = np.tile(r_in, (nt, 1))
        if np.size(z_in, 0)!= nt:
            z_in = np.tile(z_in, (nt, 1))

        nr_in = np.size(r_in, 1)
        nz_in = np.size(z_in, 1)

        if diag:
            interpBr = np.zeros((nt, nr_in),dtype='single')
            interpBz = np.zeros((nt, nr_in),dtype='single')
            interpBt = np.zeros((nt, nr_in),dtype='single')
        else:
            interpBr = np.zeros((nt, nr_in, nz_in),dtype='single')
            interpBz = np.zeros((nt, nr_in, nz_in),dtype='single')
            interpBt = np.zeros((nt, nr_in, nz_in),dtype='single')

        from scipy.constants import mu_0
        nr, nz = len(self.Rmesh), len(self.Zmesh)
        dr = (self.Rmesh[-1] - self.Rmesh[0])/(nr - 1)
        dz = (self.Zmesh[-1] - self.Zmesh[0])/(nz - 1)

        scaling = np.array([dr, dz])
        offset  = np.array([self.Rmesh[0], self.Zmesh[0]])

        unique_idx, idx =  self._get_nearest_index(tarr)
        pf = self.get_profile('PFL')
        jpol, _ = self.get_mixed('Jpol')
        Br = []
        Bt = []
        Bz = []
        for i in unique_idx:
#             psi_ax = self.psi0[i]
#             psi_sep = self.psix[i]
            sort = np.argsort(pf[i])
            jpol_spl = InterpolatedUnivariateSpline(pf[i][sort], jpol[i][sort], ext=0)
            Psi = self.pfm[...,i]
            Psi_spl = RectBivariateSpline(self.Rmesh, self.Zmesh, Psi)
            Br.append(-Psi_spl.ev(r_in, z_in, dy=1) / (2.0 * np.pi * r_in))
#             Br = -np.diff(Phi, axis=1)/(self.Rmesh[:, None]) 
#             Bz =  np.diff(Phi, axis=0)/(.5*(self.Rmesh[1:] + self.Rmesh[:-1])[:, None])
#             Bt =  np.interp(Phi, pf[i], jpol[i])*mu_0/self.Rmesh[:, None]
            Bz.append(Psi_spl.ev(r_in, z_in, dx=1) / (2.0 * np.pi * .5*(self.Rmesh[1:] + self.Rmesh[:-1])))
            Bt.append(jpol_spl(Psi_spl.ev(r_in, z_in)) * mu_0/(np.pi * r_in))
#     
#             jt = idx == i
# 
#             if diag:
#                 r = r_in[jt]
#                 z = z_in[jt]
#             else:
#                 r = np.tile(r_in[jt], (nz_in, 1, 1)).transpose(1, 2, 0)
#                 z = np.tile(z_in[jt], (nr_in, 1, 1)).transpose(1, 0, 2)
# 
#             coords =  np.array((r,z))
# 
#             index_t = ((coords.T - offset)/scaling).T
#             index_r = ((coords.T - offset)/scaling - np.array((0, .5))).T
#             index_z = ((coords.T - offset)/scaling - np.array((.5, 0))).T
# 
#             interpBr[jt] = map_coordinates(Br, index_r, mode='nearest', order=2, prefilter=True)
#             interpBz[jt] = map_coordinates(Bz, index_z, mode='nearest', order=2, prefilter=True)
#             interpBt[jt] = map_coordinates(Bt, index_t, mode='nearest', order=2, prefilter=True)

        return np.array(Br), np.array(Bz), np.array(Bt)


    def rz2rho(self, r_in, z_in, t_in=None, coord_out='rho_pol', extrapolate=True):

        """
        Equilibrium mapping routine, map from R,Z -> rho (pol,tor,r_V,...)
        Fast for a large number of points

        Input
        ----------
        t_in : float or 1darray
            time
        r_in : ndarray
            R coordinates 
            1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        z_in : ndarray
            Z coordinates 
            1D (time constant) or 2D+ (time variable) of size (nt,nx,...)
        coord_out: str
            mapped coordinates - rho_pol,  rho_tor, r_V, rho_V, Psi
        extrapolate: bool
            extrapolate coordinates (like rho_tor) for values larger than 1

        Output
        -------
        rho : 2D+ array (nt,nx,...)
        Magnetics flux coordinates of the points
        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        r_in = np.atleast_2d(r_in)
        z_in = np.atleast_2d(z_in)

        dr = (self.Rmesh[-1] - self.Rmesh[0])/(len(self.Rmesh) - 1)
        dz = (self.Zmesh[-1] - self.Zmesh[0])/(len(self.Zmesh) - 1)

        nt_in = np.size(tarr)

        if np.size(r_in, 0) == 1:
            r_in = np.tile(r_in, (nt_in, 1))
        if np.size(z_in, 0) == 1:
            z_in = np.tile(z_in, (nt_in, 1))

        if r_in.shape!= z_in.shape:
            raise Exception('Wrong shape of r_in or z_in')
        
        if np.size(r_in,0) != nt_in:
            raise Exception('Wrong shape of r_in %s'%str(r_in.shape))
        if np.size(z_in,0) != nt_in:
            raise Exception('Wrong shape of z_in %s'%str(z_in.shape))
        if np.shape(r_in) != np.shape(z_in):
            raise Exception( 'Not equal shape of z_in and r_in %s,%s'\
                            %(str(z_in.shape), str(z_in.shape)) )

        self.read_pfm()
        Psi = np.empty((nt_in,)+r_in.shape[1:], dtype=np.single)
        
        scaling = np.array([dr, dz])
        offset  = np.array([self.Rmesh[0], self.Zmesh[0]])
        
        unique_idx, idx =  self._get_nearest_index(tarr)
      
        for i in unique_idx:
            jt = idx == i
            coords = np.array((r_in[jt], z_in[jt]))
            index = ((coords.T - offset) / scaling).T
            Psi[jt] =  map_coordinates(self.pfm[:, :, i], index, mode='nearest',
                                       order=2, prefilter=True)

        rho_out = self.rho2rho(Psi, t_in=t_in, extrapolate=extrapolate, coord_in='Psi',
                              coord_out=coord_out)

        return rho_out


    def rho2rz(self, rho_in, t_in=None, coord_in='rho_pol', all_lines=False):

        """
        Get R, Z coordinates of a flux surfaces contours

        Input
        ----------

        t_in : float or 1darray
            time
        rho_in : 1darray,float
            rho coordinates of the searched flux surfaces
        coord_in: str
            mapped coordinates - rho_pol or rho_tor
        all_lines: bool:
            True - return all countours , False - return longest contour

        Output
        -------
        rho : array of lists of arrays [npoinst,2]
            list of times containg list of surfaces for different rho 
            and every surface is decribed by 2d array [R,Z]
        """

        if not self.eq_open:
            return

        if t_in is None:
            t_in = self.t_eq

        tarr  = np.atleast_1d(t_in)
        rhoin = np.atleast_1d(rho_in)

        self.read_pfm()
        self.read_scalars()

        rho_in = self.rho2rho(rhoin, t_in=t_in, \
                 coord_in=coord_in, coord_out='Psi', extrapolate=True )

        import matplotlib._cntr as cntr

        nr = len(self.Rmesh)
        nz = len(self.Zmesh)

        R, Z = np.meshgrid(self.Rmesh, self.Zmesh)
        Rsurf = np.empty(len(tarr), dtype='object')
        zsurf = np.empty(len(tarr), dtype='object')
        unique_idx, idx =  self._get_nearest_index(tarr)

        for i in unique_idx:

            jt = np.where(idx == i)[0]
            Flux = rho_in[jt[0]]

# matplotlib's contour creation
            try:
                c = cntr.Cntr(R, Z, self.pfm[:nr, :nz, i].T)
            except Exception as e:
                print(R.shape, Z.shape, self.pfm[:nr, :nz, i].T.shape)
                raise e 

            Rs_t = []
            zs_t = []

            for jfl, fl in enumerate(Flux):
                nlist = c.trace(level0=fl, level1=fl, nchunk=0)
                j_ctrs = len(nlist)//2
                if j_ctrs == 0:
                    if fl == self.psi0[i]:
                        Rs_t.append(np.atleast_1d(self.ssq['Rmag'][i]))
                        zs_t.append(np.atleast_1d(self.ssq['Zmag'][i]))
                    else:
                        Rs_t.append(np.zeros(1))
                        zs_t.append(np.zeros(1))
                    continue
                elif all_lines: # for open field lines
                    line = np.vstack(nlist[:j_ctrs])
                else:  #longest filed line 
                    line = []
                    for l in nlist[:j_ctrs]:
                        if len(l) > len(line):
                            line = l

                R_surf, z_surf = list(zip(*line))
                R_surf = np.array(R_surf, dtype = np.float32)
                z_surf = np.array(z_surf, dtype = np.float32)
                if not all_lines:
                    ind = (z_surf >= self.ssq['Zunt'][i])
                    if len(ind) > 1:
                        R_surf = R_surf[ind]
                        z_surf = z_surf[ind]
                Rs_t.append(R_surf)
                zs_t.append(z_surf)
   
            for j in jt:
                Rsurf[j] = Rs_t
                zsurf[j] = zs_t

        return Rsurf, zsurf


    def cross_surf(self, rho=1, r_in=1.65, z_in=0, theta_in=0, t_in=None, coord_in='rho_pol'):

        """
        Computes intersections of a line with any flux surface.

        Input:
        ----------
        rho: float or 1d array, size(nx)
            coordinate of the desired flux surface
        t_in: float or 1darray
            time point/array for the evaluation
        r_in: float
            R position of the point
        z_in: float
            z position of the point
        theta_in: float
            angle of the straight line with respect to horizontal-outward
        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            input coordinate label

        Output:
        ----------
        Rout: 3darray size(nt, nx, 2)
            R-position of intersections
        zout: 3darray size(nt, nx, 2)
            z-position of intersections
        """

        if not self.eq_open:
            return

        self.read_scalars()

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)
        rho  = np.atleast_1d(rho)
        r_in = np.float32(r_in)
        z_in = np.float32(z_in)

        unique_idx, idx = self._get_nearest_index(tarr)

        line_m = 3. # line length: 6 m
        n_line = int(200*line_m) + 1 # 1 cm, but then there's interpolation!
        t = np.linspace(-line_m, line_m, n_line, dtype=np.float32)

        line_r = t*np.cos(theta_in) + r_in
        line_z = t*np.sin(theta_in) + z_in
        rho_line = self.rz2rho(line_r, line_z, self.t_eq[unique_idx],
                               coord_out=coord_in, extrapolate=True)
        if coord_in == 'Psi':
            rho_line *= self.orientation
            rho      *= self.orientation
        nt_in = len(tarr)
        nrho  = len(rho)
        Rout = np.zeros((nt_in, nrho, 2), dtype=np.float32)
        zout = np.zeros((nt_in, nrho, 2), dtype=np.float32)
        for i, ii in enumerate(unique_idx):
            jt = idx == ii
            for jrho, fl in enumerate(rho):
                ind_gt = (rho_line[i] > fl)
                ind_cross = [j for j, x in enumerate(np.diff(ind_gt)) if x]
                pos = 0
                for j in ind_cross:
                    if pos > 1:
                        print('More than 2 intersections, problems')
                        zout[jt, jrho, :] = 0
                        Rout[jt, jrho, :] = 0
                        break
                    if rho_line[i, j + 1] > rho_line[i, j]:
                        ind = [j, j + 1]
                    else:
                        ind = [j + 1, j]
                    ztmp = np.interp(fl, rho_line[i, ind], line_z[ind])
                    if ztmp >= self.ssq['Zunt'][ii] and ztmp <= self.ssq['Zoben'][ii]:
                        zout[jt, jrho, pos] = ztmp
                        Rout[jt, jrho, pos] = np.interp(fl, rho_line[i, ind], line_r[ind])
                        pos += 1

        return Rout, zout


    def rhoTheta2rz(self, rho, theta_in, t_in=None, coord_in='rho_pol', n_line=201):
        
        """
        This routine calculates the coordinates R, z of the intersections of
        a ray starting at the magnetic axis with fluxsurfaces 
        at given values of some radial coordinate.
        (slower than countours)

        Input:
        ----------
        rho: float or 1D array (nr) or nD array (nt,nr,...)
            coordinate of the desired flux surface inside LCFS!
        t_in: float or 1darray
            time point/array for the evaluation

        theta_in: float or 1D array n_theta
            angle of the straight line with respect to horizontal-outward, in radians!!
            
        coord_in:  str ['rho_pol', 'rho_tor' ,'rho_V', 'r_V', 'Psi','r_a']
            input coordinate label

        Output:
        ----------
        R ,  z: 3d+ array size(nt, n_theta, nr,...)
        """
    
        if not self.eq_open:
            return

        self.read_scalars()

        if t_in is None:
            t_in = self.t_eq

        tarr = np.atleast_1d(t_in)

        nt_in = len(tarr)

        rho  = np.atleast_1d(rho)
        if rho.ndim == 1:
            rho = np.tile(rho, (nt_in, 1))

        theta_in = np.atleast_1d(theta_in)[:, None]
        ntheta = len(theta_in)

        unique_idx, idx = self._get_nearest_index(tarr)

#        n_line = 201 <=> 5 mm, but then there's interpolation!

        line_r = np.empty((len(unique_idx), ntheta, n_line))
        line_z = np.empty((len(unique_idx), ntheta, n_line))

        line_m = .9 # line length: 0.9 m
        t = np.linspace(0, 1, n_line)**.5*line_m
        c, s = np.cos(theta_in), np.sin(theta_in)
       
        tmpc = c*t
        tmps = s*t
        for i, ii in enumerate(unique_idx):
            line_r[i] = tmpc + self.ssq['Rmag'][ii]
            line_z[i] = tmps + self.ssq['Zmag'][ii]
 
        rho_line = self.rz2rho(line_r, line_z, self.t_eq[unique_idx],
                               coord_out=coord_in , extrapolate=True)

        R = np.empty((nt_in, ntheta) + rho.shape[1:], dtype='single')
        z = np.empty((nt_in, ntheta) + rho.shape[1:], dtype='single')

        if coord_in == 'Psi':
            rho_line[:,:,0] = self.psi0[unique_idx][:,None]
            rho_line *= self.orientation
            rho      *= self.orientation
        else:
            #solve some issues very close to the core
            rho_line[:,:,0] = 0

        for i, ii in enumerate(unique_idx):
            jt = idx == ii
            for k in range(ntheta):
                monotonicity = np.cumprod(np.diff(rho_line[i, k])>0)==1  #troubles with IDE
                imax = np.argmax(rho_line[i, k, monotonicity])
   
                rspl = InterpolatedUnivariateSpline(rho_line[i, k, :imax],
                                                    line_r[i, k, :imax], k=2) 
                
                R[jt, k] = rspl(rho[jt].flatten()).reshape(rho[jt].shape)

                zspl = InterpolatedUnivariateSpline( \
                       rho_line[i, k, :imax], line_z[i, k, :imax], k=2)
                z[jt, k] = zspl(rho[jt].flatten()).reshape(rho[jt].shape)

        return R, z


    def mag_theta_star(self, t_in, n_rho=400, n_theta=200, rz_grid=False ):
        
        """
        Computes theta star 

        Input:
        ----------
        t_in: float 
            time point for the evaluation
        n_rho: int
            number of flux surfaces equaly spaced from 0 to 1 of rho_pol
        n_theta: int
            number of poloidal points 
        rz_grid: bool
            evaluate theta star on the grid

        Output:
        ----------
        R, z, theta: 3d arrays size(n_rho, n_theta)
        """

        rho = np.linspace(0, 1, n_rho+1)[1:]
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

        magr, magz = self.rhoTheta2rz(rho, theta, t_in=t_in, coord_in='rho_pol')
        magr, magz = magr[0].T, magz[0].T
        
        r0 = np.interp(t_in, self.t_eq, self.ssq['Rmag'])
        z0 = np.interp(t_in, self.t_eq, self.ssq['Zmag'])

        drdrho, drtheta = np.gradient(magr)
        dzdrho, dztheta = np.gradient(magz)
        dpsidrho, dpsitheta = np.gradient(np.tile(rho**2, (n_theta, 1)).T )

        grad_rho = np.dstack((drdrho, dzdrho, dpsidrho ))
        grad_theta = np.dstack((drtheta, dztheta, dpsitheta))
        normal = np.cross(grad_rho, grad_theta, axis=-1)

        dpsi_dr = -normal[:, :, 0]/(normal[:, :, 2] + 1e-8) #Bz
        dpsi_dz = -normal[:, :, 1]/(normal[:, :, 2] + 1e-8) #Br

#WARNING not defined on the magnetics axis

        dtheta_star = ((magr - r0)**2 + (magz - z0)**2)/(dpsi_dz*(magz - z0) + dpsi_dr*(magr - r0))/magr
        theta = np.arctan2(magz - z0, - magr + r0)
        
        theta = np.unwrap(theta - theta[:, (0, )], axis=1)
        
        from scipy.integrate import cumtrapz

# Definition of the theta star by integral
        theta_star = cumtrapz(dtheta_star, theta, axis=1, initial=0)
        correction = (n_theta - 1.)/n_theta

        theta_star/= theta_star[:, (-1, )]/(2*np.pi)/correction  #normalize to 2pi

        if not rz_grid:
            return magr, magz, theta_star

# Interpolate theta star on a regular grid 
        cos_th, sin_th = np.cos(theta_star), np.sin(theta_star)
        Linterp = LinearNDInterpolator(np.c_[magr.ravel(),magz.ravel()], cos_th.ravel(),0)
             
        nx = 100
        ny = 150

        rgrid = np.linspace(magr.min(), magr.max(), nx)
        zgrid = np.linspace(magz.min(), magz.max(), ny)

        R,Z = np.meshgrid(rgrid, zgrid)
        cos_grid = Linterp(np.c_[R.ravel(), Z.ravel()]).reshape(R.shape)
        Linterp.values[:, 0] = sin_th.ravel() #trick save a some  computing time
        sin_grid = Linterp(np.c_[R.ravel(), Z.ravel()]).reshape(R.shape)  

        theta_star = np.arctan2(sin_grid, cos_grid)

        return rgrid, zgrid, theta_star


if __name__ == "__main__":

    from time import time
    import matplotlib.pylab as plt

    eqm = equ_map()
    nshot = 35662
    tref = 2.

    if eqm.Open(nshot,diag='EQH'):
        eqm.read_pfm()
        eqm.read_scalars()

        import kk
        
        R = np.linspace(1, 2.2, 1000)
        Z = np.linspace(.06, .06, 1000)
        rhop = np.linspace(0, 1, 100)

        r, z = eqm.rhoTheta2rz(rhop, 0, t_in=tref)
        out = kk.KK().kkrhorz(nshot, 3, rhop, 0)

        plt.figure(1)
        plt.plot(rhop, r[0, 0])
        plt.plot(rhop, out.r, '--')
        plt.xlabel('rho_pol')
        plt.ylabel('R')
        plt.title('kkrhorz')
        
        out = kk.KK().kkrzptfn( nshot, 3, R, Z)
        rho = eqm.rz2rho( R,Z, 3)

        plt.figure(2)
        plt.plot(R, rho[0])
        plt.plot(R, out.rho_p,'--')
        plt.xlabel('R')
        plt.ylabel('rho_pol')
        plt.title('kkrzptfn')

#------
        
        out = kk.KK().kkrhopto(nshot, 3, rhop)
        rhot = eqm.rho2rho(rhop, 3, coord_in='rho_pol', coord_out='rho_tor')

        plt.figure(3)
        plt.plot(rhop, rhot[0])
        plt.plot(rhop, out.rho_t,'--')
        plt.xlabel('rho_pol')
        plt.ylabel('rho_tor')
        plt.title('kkrhopto')
 
#------
        
        out = kk.KK().kkrzBrzt(nshot, 3, R, Z,diag='EQH')
        br, bz, bt = eqm.rz2brzt(r_in=R, z_in=Z[0], t_in=3)

        plt.figure(4, (14, 9))
        plt.subplot(1, 3, 1)
        plt.ylabel('Br')
        plt.plot(R, br[0, :, 0], 'r-', label='map_equ')
        plt.plot(R, out.br, 'b-', label='kk')
        plt.xlabel('R')
        plt.suptitle('kkrzBrzt')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(R, bz[0, :, 0], 'r-', label='map_equ')
        plt.plot(R, out.bz, 'b-', label='kk')
        plt.ylabel('Bz')
        plt.xlabel('R')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(R, bt[0, :, 0], 'r-', label='map_equ')
        plt.plot(R, out.bt, 'b-', label='kk')
        plt.ylabel('Bt')
        plt.xlabel('R')
        plt.legend()

#------

        rho = [0.1, 1]
        r2, z2 = eqm.rho2rz(rho, t_in=3, coord_in='rho_pol')
        n_theta = len(r2[0][0])
        theta = np.linspace(0, 2*np.pi, 2*n_theta)
        r1, z1 = eqm.rhoTheta2rz(rho, theta, t_in=3, coord_in='rho_pol')

        r1.shape
        plt.figure(5, figsize=(10, 12))

        plt.axes().set_aspect('equal')
        plt.plot(r2[0][0], z2[0][0], 'go')
        plt.plot(r1[0, :, 0], z1[0, :, 0], 'r+')

        plt.figure(6, figsize=(10, 12))

        plt.axes().set_aspect('equal')
        plt.plot(r2[0][1], z2[0][1], 'go')
        plt.plot(r1[0, :, 1], z1[0, :, 1], 'r+')

        plt.show()
