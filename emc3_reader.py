'''
This class reads the output written by the EMC3-Eirene code package and converts the data in SI units. 
It was written by T.Lunt. Bugs should be reported to: 
tilmann.lunt@ipp.mpg.de
Generally data files are read only once, even if the respective read-command is called twice. To avoid this behaviour force_reload should set to True.
'''

import numpy as np
from numpy import sin,cos,pi,cross
from numpy.linalg import norm,inv
import os
from scipy.interpolate import griddata
from StringIO import StringIO
import glob
import time

#some physical constants
c_vac=2.99792458e8
h_planck=6.626070040e-34
kB=1.3806488E-23
qe=1.60217657E-19
mproton=1.67262178e-27
melectron=9.10938291e-31 
gammai = 2.5
gammae  = 4.5
Eiondiss=15.6  

#typical scale settings (maybe overwritten by other script used to plot)
scls={}
scls['ne']=[0.0,3.0,1.e19,r'$n_e$ [10$^{19}$ m$^{-3}$]',False]  #min,max,scalefactor,label, logscale
scls['M']=[-1.0,1.0,1.0,r'$M$',False]
scls['Te']=[0.0,150,1.0,r'$T_e$ [eV]',False]
scls['Ti']=[0.0,150,1.0,r'$T_i$ [eV]',False]
scls['nD']=[1e14,1e19,1.0,r'$n_D$ [m$^{-3}$] (log)',True]
scls['nH']=scls['nD']
scls['gamma']=[-50,50,1.e3/qe,r'$\Gamma$ [kA/m$^2$]',False]
scls['Lc']=[0.0,100,1.0,r'$L_c$ [m]',False]
scls['nD2']=[1e14,1e19,1.0,r'$n_{D_2}$ [m$^{-3}$] (log)',True]
scls['nH2']=scls['nD2']
scls['nD+2nD2']=[1e14,1e19,1.0,r'$n_D+2n_{D_2}$ [m$^{-3}$] (log)',True]
scls['Dprof']=[0.0,1.2,1.0,r'$D$ [m$^2$/s]',False]
scls['chieprof']=[0.0,1.2,1.0,r'$\chi_e$ [m$^2$/s]',False]
scls['chiiprof']=[0.0,1.2,1.0,r'$\chi_i$ [m$^2$/s]',False]
scls['ptherm']=[0,100,1.0,r'p$_{therm}$ [Pa]',False]
scls['ptot']=[0,100,1.0,r'p$_{tot}$ [Pa]',False]
scls['Si']=[1e-2,1.e5,1.0,r'$S_i$ [A m$^{-3}$]',True]
scls['SM']=[-5.e-3,5.e-3,1.0,r'$S_M$ [kg m$^{-2}$ s$^{-2}$]',False]
scls['nZ']=[1e13,3e19,1,r'$\Sigma_i n_Z^{i+}$ [10$^{19}$ m$^{-3}$]',True]
scls['imprad']=[1.e-1,1e5,1.0,r'$P_{rad,imp}$ [W m$^{-3}$] (log)',True]
scls['PCsize']=[0.0,10,1,r'Phys. cell size',False]
scls['flux_conservation']=[1e-3,1.0,1,r'Flux conservation',True]
scls['phi']=[0,100,1.0,r'$\Phi$ [V]',False]
scls['qpar']=[1e7,5e8,1.0,r'$q_{||}$ [W/m$^2$]',False]
scls['TD']=[0,1.0,1.0,r'$T_{D}$ [eV]',False]
scls['TD2']=[0,0.1,1.0,r'$T_{D2}$ [eV]',False]
scls['pD+2pD2']=[1e0,1e4,1.0,r'$p_D+2p_{D_2}$ [Pa] (log)',True]
scls['rec_weight']=[1e-3,10,1.0,r'$n_e^2 R /\nabla \Gamma_e$ (log)',True]
scls['neut_flx']=[1e20,1.2e23,1.0,r'$|\Gamma_D| + 2|\Gamma_{D_2}|$ [m$^{-2}$ s$^{-1}$] (log)',True]
scls['R']=[1e-2,1.e5,1.0,r'$R$ [A m$^{-3}$]',True]
scls['rec_weight_HF']=[1e-3,10,1.0,r'$R/(S_i-R)$(log)',True]


#the actual reader-class
class emc3_reader():
	def __init__(self,linkfile='',iteration=0):
		self.gridpath=''
		self.Bfieldpath=''
		self.linktablepath=''
		self.equilibriumpath=''
		self.datapath={};self.loadednfield={};self.minf={};self.maxf={};self.label={}
		self.tarprofpath=''
		self.energydepopath=''
		self.particledepopath=''

		self.linkfile=linkfile
		
		if os.path.isfile(self.linkfile):
			self.lfstr=open(self.linkfile,'r').read()
			print iteration
			if iteration>0:
				self.lfstr=self.lfstr.replace('OUTPUT/','OUTPUT%i/' % iteration)
		
		self.basepath=os.path.split(self.linkfile)[0]+'/'
		self.total_volume=-1
		
			
	def findlink(self,s):
		if os.path.isfile(self.linkfile):
			fl=self.lfstr.split('\n')
			for l in fl:
				p=l.split()
				if len(p)>0:
					if (p[0]=='ln') and (p[-1]==s):
						ss=self.basepath+'/'+p[-2]
						return ss
			
		return self.basepath+'/'+s
		
	def read_grid(self,fn='',fromiz=0,toiz=-1,comp_cell_centers=True,comp_cartesian=None):
		if fn:
			fn=searchpath(fn)
		else:
			fn=self.findlink('GRID_3D_DATA')
		if not os.path.isfile(fn):
			print 'file not found ',fn
			return fn
		if fn==self.gridpath:
			print fn,' already loaded.'
			return
		f=open(fn,'r')
		self.zone=[];eof=False;iz=0;self.dataindex=0
		while (not eof) and ((iz<toiz) or (toiz<0)):
			print 'reading zone:',iz
			z=read_zone(f,self,self.dataindex);eof=eof or (not z.read_success)	
			if (z.read_success):
				if (iz>=fromiz):self.zone.append(z)
				self.dataindex+=(z.Nir-1)*(z.Nip-1)*(z.Nit-1)
				if comp_cell_centers:z.compute_cell_centers()
				if comp_cartesian:   z.compute_cartesian(comp_cartesian)
			iz+=1
		
		f.close()
		self.Niz=len(self.zone)
		self.all_read=(fromiz==0) and eof
		self.gridpath=fn
		
	def read_Bfield(self,fn=''):
		if fn:
			fn=searchpath(fn)
		else:
			fn=self.findlink('BFIELD_STRENGTH')
		if not os.path.isfile(fn):
			print 'file not found ',fn
			return fn
		if fn==self.Bfieldpath:
			print fn,' already loaded.'
			return
		f=open(fn,'r')
		if self.zone[0].offs!=0:
			print 'this function can only be called if the grid is read from the first zone.'
			return
		for iz in range(self.Niz):
			self.zone[iz].read_Bfield(f)		
		f.close()
		self.Bfieldpath=fn
		
				
	def read_data(self,name,fnlt,fnd,form,nfield=0,scaling_factor=1.0,force_reload=False):#reads EMC3-data from a given file
		fnlt=searchpath(fnlt)
		fnd =searchpath(fnd)
		if self.linktablepath!=fnlt:
			print 'loading ',fnlt
			f=open(fnlt,'r')
			nn=np.fromfile(f,dtype=int,count=3,sep=' ')
			self.Nc_geo=nn[0]; self.Nc_plas=nn[1]; self.Nc_n0=nn[2]
			
			if self.all_read:
				if self.Nc_geo != self.dataindex:
					print 'this datafield does not correspond to this grid'
					return
			self.linktable=np.fromfile(f,dtype=int,count=self.Nc_geo,sep=' ')
			f.close()
			self.linktablepath=fnlt
		else:
			print fnlt,' already loaded.'
		
		
		if not name in self.datapath.keys():
			self.datapath[name]=''
			self.loadednfield[name]=-1
		
		if (self.datapath[name]!=fnd) or (self.loadednfield[name]!=nfield) or force_reload:
			print 'loading field %s from %s (%s)' % (nfield,fnd,time.ctime(os.path.getmtime(fnd)))
			
			self.Nc_data=(self.Nc_plas if form <= 1 else self.Nc_n0)

			f=open(fnd,'r')
			for i in range(nfield+1):
				if form==0:print 'field:',np.fromfile(f,dtype=int,count=1,sep=' ')[0]
				self.Df=np.fromfile(f,dtype=float,count=self.Nc_data,sep=' ')
				#print 'last numbers',self.Df[-30:],'check',len(self.Df),self.Nc_data
				
			self.Df*=scaling_factor
			f.close()
			
			
			
			for iz in range(self.Niz):
				print 'loading zone',iz
				self.zone[iz].decode_data(name)
				
			self.minf[name]=np.min([self.zone[iz].minf[name] for iz in range(self.Niz)]) #the global minimum
			self.maxf[name]=np.max([self.zone[iz].maxf[name] for iz in range(self.Niz)]) #the global maximum			
			print 'done.'
			self.datapath[name]=fnd;self.loadednfield[name]=nfield
			
		else:
			print 'field ',nfield,'from',fnd,' already loaded.'
	
	def read_intersection(self,fn=''):
		if fn:
			fn=searchpath(fn)
		else:
			fn=self.findlink('PLATES_MAG')
		if not os.path.isfile(fn):
			print 'file not found ',fn
			return fn

		name='intersection'	
		if not name in self.datapath.keys():
			self.datapath[name]=''
			self.loadednfield[name]=-1
			
		if fn==self.datapath[name]:
			print fn,' already loaded.'
			return

		for zn in self.zone:
			zn.field[name]=np.zeros((zn.Nir-1,zn.Nip-1,zn.Nit-1))
			zn.minf[name]=0;zn.maxf[name]=1;
			zn.findvalidboundaries(name=name)

		for l in open(fn).read().split('\n')[:-1]:
			ii=np.fromstring(l,dtype=int,sep=' ')
			#print ii
			iz,ir,ip=ii[0:3]
			for i in range(ii[3]/2):
				it=np.arange(ii[4+2*i],ii[5+2*i]+1)
				self.zone[iz].field[name][ir,ip,it]=1		
				
				#with open(fn, 'r') as f: 
					#for s in f:
						#a=np.fromstring(s,dtype=int,count=4,sep=' ')
						#n=np.fromstring(s,dtype=int,count=4+a[3],sep=' ')
						#for i in range(0,a[3],2):
							#self.zone[n[0]].field[name][n[1],n[2],n[4+i]:n[5+i]]=1


		self.minf[name]=np.min([self.zone[iz].minf[name] for iz in range(self.Niz)]) #the global minimum
		self.maxf[name]=np.max([self.zone[iz].maxf[name] for iz in range(self.Niz)]) #the global maximum			
		print 'done.'
		self.datapath[name]=fn
		self.label[name]='intersection'
				
	
	def read_fields(self,what,amass=2,zrange=[0]): # this routines reads the datafields of EMC3-Eirene. It only works if standard links (fort.31 -> DENSITY, ...) are defined.
		if not isinstance(what, list):
			what=[what]
			
		for w in what:
			print 'loading',w,self.basepath
			
			
			if w=='ne':
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.31'),0,nfield=0,scaling_factor=1.e6)
				self.label[w]=r'n$_e$ [m$^{-3}$]'

			elif w=='M':
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.33'),1,nfield=0)
				self.label[w]=r'M'
				
			elif w=='Te':
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=0)
				self.label[w]=r'T$_e$ [eV]'

			elif w=='Ti':
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=1)
				self.label[w]=r'T$_i$ [eV]'
				
			elif w=='Lc':
				self.read_data(w,self.findlink('fort.70'),self.findlink('CONNECTION_LENGTH'),1,nfield=0,scaling_factor=1.e-2)
				self.label[w]=r'$L_c$ [m]'
								
			elif (w=='nH') or (w=='nD'):
				self.read_data(w,self.findlink('fort.70'),self.findlink('DENSITY_A'),2,nfield=0,scaling_factor=1.e6)
				self.label[w]=r'n$_D$ [m$^{-3}$]'
				
			elif (w=='nH2') or (w=='nD2'):
				self.read_data(w,self.findlink('fort.70'),self.findlink('DENSITY_M'),2,nfield=0,scaling_factor=1.e6)
				self.label[w]=r'n$_{H_2}$ [m$^{-3}$]'
				
			elif (w=='nH+2nH2'):	
				self.read_data('nH' ,self.findlink('fort.70'),self.findlink('DENSITY_A'),2,nfield=0,scaling_factor=1.e6)
				self.read_data('nH2',self.findlink('fort.70'),self.findlink('DENSITY_M'),2,nfield=0,scaling_factor=1.e6)
				for zn in self.zone:
					zn.make_empty_field(w)
					zn.field[w]=zn.field['nH']+2*zn.field['nH2']
				
				self.label[w]=r'$n_H+2n_{H_2}$ [m$^{-3}$]'
				
			elif (w=='nD+2nD2'):	
				self.read_data('nD' ,self.findlink('fort.70'),self.findlink('DENSITY_A'),2,nfield=0,scaling_factor=1.e6)
				self.read_data('nD2',self.findlink('fort.70'),self.findlink('DENSITY_M'),2,nfield=0,scaling_factor=1.e6)
				for zn in self.zone:
					zn.make_empty_field(w)
					zn.field[w]=zn.field['nD']+2*zn.field['nD2']
				
				self.label[w]=r'$n_D+2n_{D_2}$ [m$^{-3}$]'				

			elif (w=='neut_flx'):	
				self.read_data('nD' ,self.findlink('fort.70'),self.findlink('DENSITY_A'),2,nfield=0,scaling_factor=1.e6)
				self.read_data('nD2',self.findlink('fort.70'),self.findlink('DENSITY_M'),2,nfield=0,scaling_factor=1.e6)
				self.read_data('TD',self.findlink('fort.70'),self.findlink('TEMPERATURE_A'),2,nfield=0)
				self.read_data('TD2',self.findlink('fort.70'),self.findlink('TEMPERATURE_M'),2,nfield=0)
				
				for zn in self.zone:
					zn.make_empty_field(w)
					#cf formula 2.21 and 2.24 in stangeby's book
					zn.field[w]=0.25*zn.field['nD']*(8*zn.field['TD']*qe/(np.pi*2*mproton))**0.5 + 2*0.25*zn.field['nD2']*(8*zn.field['TD2']*qe/(np.pi*4*mproton))**0.5  
				
				self.label[w]=r'$|\Gamma_D| + 2|\Gamma_{D_2}|$ [m$^{-2}$ s$^{-1}$]'				


			elif (w=='TH') or (w=='TD'):
				self.read_data(w,self.findlink('fort.70'),self.findlink('TEMPERATURE_A'),2,nfield=0)
				self.label[w]=r'T$_{'+w[1]+'}$ [m$^{-3}$]'

			elif (w=='TH2') or (w=='TD2'):
				self.read_data(w,self.findlink('fort.70'),self.findlink('TEMPERATURE_M'),2,nfield=0)
				self.label[w]=r'T$_{'+w[1]+'2}$ [m$^{-3}$]'
				
			elif (w=='pD+2pD2'):

				self.read_data('nD' ,self.findlink('fort.70'),self.findlink('DENSITY_A'),2,nfield=0,scaling_factor=1.e6)
				self.read_data('nD2',self.findlink('fort.70'),self.findlink('DENSITY_M'),2,nfield=0,scaling_factor=1.e6)

				self.read_data('TD' ,self.findlink('fort.70'),self.findlink('TEMPERATURE_A'),2,nfield=0)
				self.read_data('TD2',self.findlink('fort.70'),self.findlink('TEMPERATURE_M'),2,nfield=0)
				for zn in self.zone:
					zn.make_empty_field(w)
					zn.field[w]=(zn.field['nD']*zn.field['TD']+2*zn.field['nD2']*zn.field['TD2'])*qe
				
				self.label[w]=r'$p_D+2p_{D_2}$ [Pa]'				
				

			elif w in ['Dprof','chieprof','chiiprof']:
				fninppar=self.findlink('fort.2')
				print 'reading: ',fninppar
				
				ll=[l for l in open(fninppar).read().split('\n') if not l.startswith('*')]
				n=np.fromstring(ll[0],dtype=int,sep=' ')[0]
				nnD=np.fromstring(ll[n+1],dtype=float,sep=' ',count=1)[0]
				nnchi=np.fromstring(ll[2*n+1],dtype=float,sep=' ',count=2)

				print 'D',nnD,nnchi
				nn={'Dprof':nnD,'chieprof':nnchi[0],'chiiprof':nnchi[1]}[w]

				if nn>=0.0:
					print 'constant %s: %.2f m^2/s' % (w,nn/1e4)
					for iz in range(self.Niz):
						zn=self.zone[iz]
						zn.make_empty_field(w)
						zn.field[w][:,:,:]=nn/1e4
				else:
					fnperp=self.findlink('fort.'+str(int(-nn)))
					print 'reading %s from %s ' % (w,fnperp)
					
					self.read_data(w,self.findlink('fort.70'),fnperp,1,nfield=0,scaling_factor=1.e-4)
					self.label[w]=r'$%s$ [m$^2$/s]' % {'Dprof':'D_perp','chieprof':'\chi_{e,\perp}','chiiprof':'\chi_{i,\perp}'}[w]
					

			elif w=='phi':
				self.read_data(w,self.findlink('fort.70'),self.basepath+'POTENTIAL',1,nfield=0)
				self.label[w]=r'$\Phi$ [V]'

			elif w=='current':
				self.read_data(w,self.findlink('fort.70'),self.basepath+'CURRENT_V',1,nfield=0)
				self.label[w]=r'$j$ [A]'

			elif w=='res':
				self.read_data(w,self.findlink('fort.70'),self.basepath+'RESISTIVITY',1,nfield=0)
				self.label[w]=r'$\int \sigma_{||}^{-1}dl$ [$\Omega$ cm$^2$]'

				
			elif (w=='ptherm') or (w=='ptot') or (w=='gamma') or (w=='qpar'):
				self.read_data('ne',self.findlink('fort.70'),self.findlink('fort.31'),0,nfield=0,scaling_factor=1.e6)
				self.read_data('Te',self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=0)
				self.read_data('Ti',self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=1)
				mi=amass*mproton				
				if w=='ptherm':
					for zn in self.zone:
						zn.field[w]=zn.field['ne']*(zn.field['Te']+zn.field['Ti'])*qe
				elif w=='ptot':
					self.read_data('M' ,self.findlink('fort.70'),self.findlink('fort.33'),1,nfield=0)				
					for zn in self.zone:
						zn.field[w]=zn.field['ne']*(zn.field['Te']+zn.field['Ti'])*(1+zn.field['M']**2)*qe
					self.label[w]=r'p$_{tot}$ [Pa]'		
				elif w=='gamma':
					self.read_data('M',self.findlink('fort.70'),self.findlink('fort.33'),1,nfield=0)
					for zn in self.zone:
						zn.field[w]=zn.field['ne']*zn.field['M']*((zn.field['Te']+zn.field['Ti'])*qe/mi)**0.5
					self.label[w]=r'$\Gamma$ [A/m$^2$]'							
				elif w=='qpar':

					for zn in self.zone:
						flux=0.5*zn.field['ne']*((zn.field['Te']+zn.field['Ti'])*qe/mi)**0.5
						zn.field[w]=flux*qe*(gammae*zn.field['Te']+gammai*zn.field['Ti']+Eiondiss)					
				for zn in self.zone:
					zn.minf[w]=np.nanmin(zn.field[w])
					zn.maxf[w]=np.nanmax(zn.field[w])					
					zn.findvalidboundaries(w)
				self.minf[w]=np.min([self.zone[iz].minf[w] for iz in range(self.Niz)]) #the global minimum
				self.maxf[w]=np.max([self.zone[iz].maxf[w] for iz in range(self.Niz)]) #the global maximum


			elif w=='Si':
				nn=np.loadtxt(self.basepath+'/OUTPUT/RECYC_FLUX')
				print 'RECYC_FLUX=',nn
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.40'),1,nfield=0,scaling_factor=1e6*nn[0])				
				self.label[w]=r'$S_i$ [A m$^{-3}$]'

			elif w=='SM':
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.47'),1,nfield=0,scaling_factor=1.e4/qe)				
				self.label[w]=r'$S_M$ [kg m$^{-2}$ s$^{-2}$]'
					

			elif w=='nZ':
				print 'loading nz'
				self.read_data(w,self.findlink('fort.70'),self.findlink('fort.31'),0,nfield=0,scaling_factor=0.0)
				for i in zrange:
					ww=w+'%03i'%i
					print 'charge state',ww,'+'
					if i==0:
						self.read_data(ww,self.findlink('fort.70'),self.findlink('IMPURITY_NEUTRAL'),0,nfield=i,scaling_factor=1.e6)					
					else:
						self.read_data(ww,self.findlink('fort.70'),self.findlink('fort.31'),0,nfield=i,scaling_factor=1.e6)
					for zn in self.zone:zn.field[w]+=zn.field[ww]
				self.loadednfield[w]=-1
				self.label[w]=r'$\Sigma$ n$_Z$ [m$^{-3}$]'

			elif w=='imprad':
				print 'loading impurity radiaton'
				self.read_data(w,self.findlink('fort.70'),self.findlink('IMP_RADIATION'),1,nfield=0,scaling_factor=-1.0e6)
				self.label[w]=r'$P_{rad,imp}$ [W m$^{-3}$]'

			elif w=='Simp':
				print 'loading impurity ionization source'
				self.read_data(w,self.findlink('fort.70'),self.findlink('IMPURITY_IONIZATION_SOURCE'),1,nfield=0)
				self.label[w]=r'S$_Z$ []'
					
					
			elif w=='PCsize':
				self.read_data(w,self.findlink('fort.70'),'',0,nfield=-1)
				count=np.zeros(self.Nc_plas)
				for iz in range(self.Niz):
					zn=self.zone[iz]
					Nr=zn.Nir-1;Np=zn.Nip-1;Nt=zn.Nit-1
					D=zn.field[w]=np.zeros((Nr,Np,Nt))								
					
					for ir in range(Nr):
						for ip in range(Np):
							for it in range(Nt):
								il=it*Np*Nr + ip*Nr + ir +zn.offs
								ic=self.linktable[il]-1
								if (ic >= 0) and (ic < self.Nc_plas):count[ic]+=1
					for ir in range(Nr):
						for ip in range(Np):
							for it in range(Nt):
								il=it*Np*Nr + ip*Nr + ir +zn.offs
								ic=self.linktable[il]-1
								if (ic >= 0) and (ic < self.Nc_plas):D[ir,ip,it]=count[ic]

				
				for zn in self.zone:
					zn.minf[w]=np.nanmin(zn.field[w])
					zn.maxf[w]=np.nanmax(zn.field[w])					
					zn.findvalidboundaries(w)
				self.minf[w]=np.min([self.zone[iz].minf[w] for iz in range(self.Niz)]) #the global minimum
				self.maxf[w]=np.max([self.zone[iz].maxf[w] for iz in range(self.Niz)]) #the global maximum

				self.label[w]=r'physical cell size'
			elif w=='flux_conservation':
				print self.findlink('fort.70')
				print self.findlink('FLUX_CONSERVATION')
				self.read_data(w,self.findlink('fort.70'),self.findlink('FLUX_CONSERVATION'),1,nfield=0)
				self.label[w]=r'flux []'
				
			elif (w=='rec_weight'):
				self.read_data('ne',self.findlink('fort.70'),self.findlink('fort.31'),0,nfield=0,scaling_factor=1.e6)
				self.read_data('Te',self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=0)
				self.read_data('Ti',self.findlink('fort.70'),self.findlink('fort.30'),1,nfield=1)
				self.read_data('M',self.findlink('fort.70'),self.findlink('fort.33'),1,nfield=0)
				self.read_data('Lc',self.findlink('fort.70'),self.findlink('CONNECTION_LENGTH'),1,nfield=0,scaling_factor=1.e-2)
				self.read_data('ne_ne_R',self.findlink('fort.70'),'D_rec_rate.txt',1,nfield=0)
				
				mi=amass*mproton
				
				for zn in self.zone:
					zn.make_empty_field('gamma')
					zn.field['gamma']=zn.field['ne']*zn.field['M']*((zn.field['Te']+zn.field['Ti'])*qe/mi)**0.5
					
					zn.make_empty_field(w)
					zn.field[w]=zn.field['ne_ne_R']*zn.field['Lc']/np.abs(zn.field['gamma'])
					zn.field[w][np.where(zn.field['Lc']>500)]=np.nan
				
				self.label[w]=r'$n_e^2 R / \nabla \Gamma_e$'				
			elif w=='R':
				self.read_data(w,self.findlink('fort.70'),self.findlink('RECOMBINATION'),1,nfield=0,scaling_factor=1.e6*qe)
				self.label[w]=r'R [s$^{-1}$ m$^{-3}$]'

			elif (w=='rec_weight_HF'):
				self.read_data('R',self.findlink('fort.70'),self.findlink('RECOMBINATION'),1,nfield=0,scaling_factor=1.e6*qe)
				for zn in self.zone:
					zn.make_empty_field(w)
					zn.field[w]=np.abs(zn.field['R'])/np.abs(zn.field['Si'])
				
				self.label[w]=r'R/(Si-R)'

							
			else:
				print 'unknown field: ',w





	def writeout_plasma_field(self,name,fnd,scaling_factor=1.0,form=0):
		self.Df=np.zeros(self.Nc_plas)
		self.Ndf=np.zeros(self.Nc_plas,dtype=int)
		self.Nc_data=self.Nc_plas if form==0 else self.Nc_n0
		for iz in range(self.Niz):
			print 'encoding zone',iz
			self.zone[iz].encode_data(name)
		self.Df/=self.Ndf
		self.Df*=scaling_factor
		f=open(fnd,'w') 
		np.savetxt(f,self.Df,fmt='%.5e')
		f.close()

	
	def read_target_profiles(self,fn=None,sep=None,fromitar=0,toitar=-1,appendtargets=False):

		if fn==None:
			fname=self.findlink('TARGET_PROFILES')
		else:
			fname=searchpath(self.basepath+fn+';'+fn)

		if self.tarprofpath!=fname:
			print 'fname=%s (%s)' %(fname,time.ctime(os.path.getmtime(fname)))
			f=open(fname,'r')
			nrt=np.fromfile(f,dtype=int,count=1,sep=' ')[0]
			print nrt,sep
			itar=0	
			if (not appendtargets): self.tarprof=[]
			while (itar<nrt) and ((itar<toitar) or (toitar<0)):
				t=read_target_profile(f,sep)
				if itar>=fromitar:
					self.tarprof.append(t)
				itar+=1		
			f.close()
			self.tarprofpath=fname
		else:
			print fname,' already loaded.'
	
	def read_energy_depo(self,identifier=None,fixbug=False):
		'''comment on fixbug: fortran converts numbers smaller than 1E-99 into 1-100 or simliar.
		set fixbug=True to ignore these numbers.'''
		if identifier==None:
			fn=self.findlink('ENERGY_DEPO')
		elif isinstance(identifier,str):
			fn=searchpath(self.basepath+identifier+';'+identifier)
		else:
			print 'unknown identifier'
			return
		if self.energydepopath!=fn:
			print 'fn=',fn
			s=open(fn,'r').read()
			if fixbug:
				s='\n'.join([line for line in s.split('\n')[:-1] if line[-4]!='-'])
			
			self.energydepo=np.loadtxt(StringIO(s.split("MAPPING")[0]),skiprows=1,dtype=float)
			self.energymapp=np.loadtxt(StringIO(s.split("MAPPING")[1]),skiprows=0,dtype=float)
			self.energydepopath=fn

	
	def read_particle_depo(self,identifier=None,fixbug=False):
		'''comment on fixbug: fortran converts numbers smaller than 1E-99 into 1-100 or simliar.
		set fixbug=True to ignore these numbers.'''	
		if identifier==None:
			fn=self.findlink('PARTICLE_DEPO')
		elif isinstance(identifier,str):
			fn=searchpath(self.basepath+identifier+';'+identifier)
		else:
			print 'unknown identifier'
			return
		if self.particledepopath!=fn:
			print 'fn=',fn
			s=open(fn,'r').read()
			if fixbug:
				s='\n'.join([line for line in s.split('\n')[:-1] if line[-4]!='-'])

			self.particledepo=np.loadtxt(StringIO(s.split("MAPPING")[0]),skiprows=1,dtype=float)
			self.particlemapp=np.loadtxt(StringIO(s.split("MAPPING")[1]),skiprows=0,dtype=float)
			
	
	
	def compute_cell_centers(self):
		for zn in self.zone:zn.compute_cell_centers()

	def compute_cartesian(self,form='x_y_z'):
		for zn in self.zone:zn.compute_cartesian(form)	

	def compute_volume_and_area(self,force_computation=False):
		for iz in range(self.Niz):
			print 'computing zone:',iz
			self.zone[iz].compute_volume_and_area(force_computation)
		self.total_volume=np.sum([z.total_volume for z in self.zone])
	
	def interpol_plane(self,phi,Rzlist,names,method='linear'):
		phi=(phi+360.) % 360.
		RR=[rz[0] for rz in Rzlist];zz=[rz[1] for rz in Rzlist]
		minR=min(RR);maxR=max(RR)
		minz=min(zz);maxz=max(zz)
		
		if abs(minR-maxR)<0.2:
			mR=(minR+maxR)*0.5;minR=mR-0.1;maxR=mR+0.1
				
		if abs(minz-maxz)<0.2:
			mz=(minz+maxz)*0.5;minz=mz-0.1;maxz=mz+0.1
			
		points=[];values={}
		for name in names:values[name]=[]
		for iz in range(self.Niz):
			zn=self.zone[iz]
			for it in range(zn.Nit-1):
				x=(phi-zn.phi[it])/(zn.phi[it+1]-zn.phi[it])				
				if (x>=0) and (x<1.0):					
					print 'plane found',iz,it
					for ir in range(zn.Nir-1):
						for ip in range(zn.Nip-1):
#							if (zn.Rc[ir,ip,it]>=minR) and (zn.Rc[ir,ip,it]<=maxR) and (zn.zc[ir,ip,it]>=minz) and (zn.zc[ir,ip,it]<=maxz):
							points.append((zn.Rc[ir,ip,it],zn.zc[ir,ip,it]))
							for name in names:
								values[name].append(zn.field[name][ir,ip,it])
		
		profile={}
		for name in names:
			print len(points),len(values[name]),len(Rzlist)
			profile[name]=griddata(points, values[name], Rzlist, method=method)
		
		return profile
			
	def prepare_frequent_interpolation(self,names):
		self.points=points=[];self.values=values={}
		for name in names:values[name]=[]
		for iz in range(self.Niz):
			zn=self.zone[iz]
			it=8
			for ir in range(zn.Nir-1):
				for ip in range(zn.Nip-1):
#					if (zn.Rc[ir,ip,it]>=minR) and (zn.Rc[ir,ip,it]<=maxR) and (zn.zc[ir,ip,it]>=minz) and (zn.zc[ir,ip,it]<=maxz):
					points.append((zn.Rc[ir,ip,it],zn.zc[ir,ip,it]))
					for name in names:
						values[name].append(zn.field[name][ir,ip,it])
			
		points = np.asanyarray(points)
		
		
	def frequent_interpol_plane(self,Rzlist,names,method='linear'):
		profile={}
		for name in names:
			#print 'len(points)',points.shape,'len(values[name])',len(values[name]),'len(Rzlist)',len(Rzlist) #,'points.shape[1]',points.shape[1]
			#print 'name',name
			profile[name]=griddata(self.points, self.values[name], Rzlist, method=method)
		
		return profile
		
		
		
	def get_separatrix(self,iz):
		zn=self.zone[iz];it=(zn.Nit-1)/2
		return np.array([(zn.R[0,ip,it],zn.z[0,ip,it]) for ip in range(zn.Nip)])
			

	def get_omp(self, izc,izs,it,N=200):#gives back a list of Rz values at the outboard mid-plane
		core=self.zone[izc];sol=self.zone[izs]
		ipc=max((v, i) for i, v in enumerate(core.R[-1,:,it]))[1] #gives back the index where R is maximum
		
		ips=[ip for ip in range(sol.Nip) if (sol.R[0,ip,it]==core.R[-1,ipc,it]) and (sol.z[0,ip,it]==core.z[-1,ipc,it])][0] #gives back the corresponding index in the SOL
		
		Rsep=core.R[-1,ipc,it];zsep=core.z[-1,ipc,it]
		
		p1=np.array([core.R[0,ipc,it],zsep])
		p2=np.array([sol.R[-1,ips,it],zsep])

		Rz=vectorlist(p1,p2,N)
		
		return Rz,Rsep,ipc,ips
			
	def get_op_xp(self,izc):
		c=self.zone[izc];it=(c.Nit-1)/2;ip=(c.Nip-1)/4;ip2=(c.Nip-1)/2
		p1=np.array([c.R[-1, 0,it],c.z[-1, 0,it]]);p2=np.array([c.R[0, 0,it],c.z[0, 0,it]]);p2=p1+(p2-p1)*10
		q1=np.array([c.R[-1,ip,it],c.z[-1,ip,it]]);q2=np.array([c.R[0,ip,it],c.z[0,ip,it]]);q2=q1+(q2-q1)*10
		r1=np.array([c.R[-1,ip2,it],c.z[-1,ip2,it]]);r2=np.array([c.R[0,ip2,it],c.z[0,ip2,it]]);r2=r1+(r2-r1)*10
		
		llc=line_line_col(p1,p2,q1,q2);llc2=line_line_col(r1,r2,q1,q2)
		op=p1+(p2-p1)*llc[0];op2=r1+(r2-r1)*llc2[0]
		if np.linalg.norm(op-op2)<1e-7:
			return p1+(p2-p1)*llc[0],p1
		else:
			print 'O-point not found!'
			return None
		
	def read_equilibrium(self,fn,izcore=None,oxp=None,oxpfn=None,gen_datafield=False):
		fn=searchpath(self.basepath+fn+';'+fn)
		self.equilibriumpath=fn
		name='Psi'		

		if '.xml' in fn:
			import xml.etree.ElementTree as et
			def xml2np(pxml):
				N=np.fromstring(pxml.attrib['dim'],dtype=int,sep=',')
				return np.fromstring(pxml.text,sep=',',dtype=float,count=np.product(N)).reshape(N)

			magneticsxml=et.parse(fn).getroot()
			eqxml= magneticsxml.find('equilibrium') 
			self.Requi=xml2np(eqxml.find('Ri'))
			self.zequi=xml2np(eqxml.find('zj'))
			self.psi=xml2np(eqxml.find('PFM'))
			singxml=eqxml.find('singularities')
  
			if singxml is not None:
				self.op=op=xml2np(singxml.find('op'))
				self.xp=xp=xml2np(singxml.find('xp'))
				self.xp2=xp2=xml2np(singxml.find('xp2'))
				self.psi=(self.psi-op[0])/(xp[0]-op[0])
				name='PsiN'	
		else:
			f=open(fn,'r')
			n=np.fromstring(f.readline(),dtype=int,count=2,sep=' ')
			lim=np.fromstring(f.readline(),dtype=float,count=4,sep=' ')
			self.Requi=np.linspace(lim[0],lim[1],n[0])
			self.zequi=np.linspace(lim[2],lim[3],n[1])
			self.psi=np.fromfile(f,dtype=float,count=n[0]*n[1],sep=' ').reshape(n[1],n[0])
			f.close()
		
		
		

			if (oxp!=None):
				f=self.get_psi([oxp[0],oxp[1]])
				self.psi=(self.psi-f[0])/(f[1]-f[0])
				name='PsiN'				
			if (oxpfn!=None):
				fn=searchpath(self.basepath+oxpfn+';'+oxpfn)
				f=open(fn,'r')
				s=f.read()
				f.close()
				header='2.  *******> O- and X-points'
				if not (header in s):
					print 'no O- and X-point positions found'
					return()
				self.xp=[]
				for line in s.split('\n'):
					if line.startswith('O-point:'):self.op=op=np.fromstring(line[9:],sep=' ',dtype=float);op[1:3]/=100.
					if line.startswith('X-point:'):self.xp.append(np.fromstring(line[9:],sep=' ',dtype=float));self.xp[-1][1:3]/=100.
				xp=self.xp[0]

				print 'op,xp=',op,xp
				self.psi=(self.psi-op[0])/(xp[0]-op[0])
				name='PsiN'									
			elif (izcore!=None):
				op,xp=self.get_op_xp(izcore)
				f=self.get_psi([op,xp])
				self.psi=(self.psi-f[0])/(f[1]-f[0])
				name='PsiN'
		if gen_datafield:	
			for iz in range(self.Niz):
				print iz
				zn=self.zone[iz]
				zn.field[name]=np.zeros((zn.Nir-1,zn.Nip-1,zn.Nit-1))
				N=(zn.Nir-1)*(zn.Nip-1)*(zn.Nit-1)
				x=zn.Rc.reshape(N)
				y=zn.zc.reshape(N)
				Rz=[(x[i],y[i]) for i in range(N)]
				zn.field[name]=self.get_psi(Rz).reshape((zn.Nir-1,zn.Nip-1,zn.Nit-1))
				zn.firstir[name]=0;zn.lastir[name]=zn.Nir-1
				zn.firstip[name]=0;zn.lastip[name]=zn.Nip-1
		

		
	def get_psi(self,Rz,method='cubic'):
		X, Y = np.meshgrid(self.Requi, self.zequi)
		N=len(self.Requi)*len(self.zequi);X=X.reshape(N);Y=Y.reshape(N)
		points=[(X[i],Y[i]) for i in range(N)]
		return griddata(points, self.psi.reshape(N), Rz,method=method)
		
	def magnetic_coordinates(self,izcore):
		op,xp=self.get_op_xp(izcore)
		X, Y = np.meshgrid(self.Requi, self.zequi)
		N=len(self.Requi)*len(self.zequi);X=X.reshape(N);Y=Y.reshape(N)
		points=[(X[i],Y[i]) for i in range(N)]
		
		for zn in self.zone:
			Nr=zn.Nir-1;Np=zn.Nip-1;Nt=zn.Nit-1
			zn.rhoc=np.zeros((Nr,Np,Nt));zn.thetac=np.zeros((Nr,Np,Nt))
			Rz=[]
			for ir in range(Nr):
				for ip in range(Np):
					for it in range(Nt):
						x=[zn.Rc[ir,ip,it],zn.zc[ir,ip,it]]
						Rz.append(x)
						zn.thetac[ir,ip,it]=np.arctan2(x[1]-op[1],x[0]-op[0])
			zn.rhoc=griddata(points, self.psi.reshape(N), Rz).reshape(Nr,Np,Nt)**0.5						

class read_zone():
	def __init__(self,f,parent,dataoffset):	
		nn=np.fromfile(f,dtype=int,count=3,sep=' ')
		self.read_success=False
		if len(nn) !=3:return
		Nir=self.Nir=nn[0];Nip=self.Nip=nn[1];Nit=self.Nit=nn[2]
		print Nir,Nip,Nit
		self.phi=np.zeros(Nit)
		self.R=np.zeros((Nir,Nip,Nit))
		self.z=np.zeros((Nir,Nip,Nit))
		self.field={};self.minf={};self.maxf={};self.firstir={};self.lastir={};self.firstip={};self.lastip={}
		self.offs=dataoffset
		self.parent=parent
		self.volumepath=''
		for it in range(Nit):
			self.phi[it]=np.fromfile(f,dtype=float,count=1,sep=' ')[0]#+360.) % 360.
			self.R[:,:,it]=np.fromfile(f,dtype=float,count=Nir*Nip,sep=' ').reshape(Nip,Nir).T*0.01
			self.z[:,:,it]=np.fromfile(f,dtype=float,count=Nir*Nip,sep=' ').reshape(Nip,Nir).T*0.01
		self.read_success=True

	def compute_cartesian(self,form='x_y_z'):
		if 'x_y_z' in form:
			x=self.x=np.zeros((self.Nir,self.Nip,self.Nit))
			y=self.y=np.zeros((self.Nir,self.Nip,self.Nit))
			for it in range(self.Nit):
				x[:,:,it]=self.R[:,:,it]*np.cos(self.phi[it]*np.pi/180.)
				y[:,:,it]=self.R[:,:,it]*np.sin(self.phi[it]*np.pi/180.)

		if 'xyz' in form:
			xyz=self.xyz=np.zeros((self.Nir,self.Nip,self.Nit,3))			
			for it in range(self.Nit):
				xyz[:,:,it,0]=self.R[:,:,it]*np.cos(self.phi[it]*np.pi/180.)
				xyz[:,:,it,1]=self.R[:,:,it]*np.sin(self.phi[it]*np.pi/180.)
			xyz[:,:,:,2]=self.z	
			
		if 'xc_yc_zc' in form:	
			xc=self.xc=np.zeros((self.Nir-1,self.Nip-1,self.Nit-1))
			yc=self.yc=np.zeros((self.Nir-1,self.Nip-1,self.Nit-1))
			for it in range(self.Nit-1):
				xc[:,:,it]=self.Rc[:,:,it]*np.cos(self.phic[it]*np.pi/180.)
				yc[:,:,it]=self.Rc[:,:,it]*np.sin(self.phic[it]*np.pi/180.)			
				
		if 'xcyczc' in form:	
			xcyczc=self.xcyczc=np.zeros((self.Nir-1,self.Nip-1,self.Nit-1,3))
			for it in range(self.Nit-1):
				xcyczc[:,:,it,0]=self.Rc[:,:,it]*np.cos(self.phic[it]*np.pi/180.)
				xcyczc[:,:,it,1]=self.Rc[:,:,it]*np.sin(self.phic[it]*np.pi/180.)
			xcyczc[:,:,:,2]=self.zc
				
	def read_Bfield(self,f):	
		self.B=np.fromfile(f,dtype=float,count=self.Nir*self.Nip*self.Nit,sep=' ').reshape(self.Nit,self.Nip,self.Nir).T
		self.Bc=0.5*(self.B[0:-1,0:-1,0:-1]+self.B[1:,1:,1:])

	def make_empty_field(self,name):
		Nr=self.Nir-1;Np=self.Nip-1;Nt=self.Nit-1		
		D=np.empty((Nr,Np,Nt))
		D[:,:,:]=np.NAN
		self.field[name]=D
		self.minf[name]=0.0
		self.maxf[name]=1.0
		self.firstir[name]=0;self.lastir[name]=Nr
		self.firstip[name]=0;self.lastip[name]=Np
		
				
	def decode_data(self,name='D'):
		Nr=self.Nir-1;Np=self.Nip-1;Nt=self.Nit-1
		
		linktable=self.parent.linktable;Df=self.parent.Df
		D=np.empty((Nr,Np,Nt))
		D[:,:,:]=np.NAN
		#print 'Nc_data=',self.parent.Nc_data,len(Df)
		for ir in range(Nr):
			for ip in range(Np):
				for it in range(Nt):
					il=it*Np*Nr + ip*Nr + ir +self.offs
					ic=linktable[il]- 1
					if (ic >= 0) and (ic < self.parent.Nc_data): D[ir,ip,it]=Df[ic]
		self.field[name]=D
		self.minf[name]=np.nanmin(D)
		self.maxf[name]=np.nanmax(D)
		self.findvalidboundaries(name=name)
		
	def findvalidboundaries(self,name='D'):
		Nr=self.Nir-1;Np=self.Nip-1;Nt=self.Nit-1
		D=self.field[name]
		ir=0		
		while np.isnan(np.nanmin(D[ir,:,:])) and (ir<Nr):ir+=1
		self.firstir[name]=ir
		ir=Nr-1		
		while np.isnan(np.nanmin(D[ir,:,:])) and (ir>0):ir-=1
		self.lastir[name]=ir		
		ip=0		
		while np.isnan(np.nanmin(D[:,ip,:])) and (ip<Np):ip+=1
		self.firstip[name]=ip
		ip=Np-1		
		while np.isnan(np.nanmin(D[:,ip,:])) and (ip>0):ip-=1
		self.lastip[name]=ip		

	def encode_data(self,name='D'):
		Nr=self.Nir-1;Np=self.Nip-1;Nt=self.Nit-1
		
		linktable=self.parent.linktable;Df=self.parent.Df;Ndf=self.parent.Ndf
		D=self.field[name]

		for ir in range(Nr):
			for ip in range(Np):
				for it in range(Nt):
					il=it*Np*Nr + ip*Nr + ir +self.offs
					ic=linktable[il]- 1
					if (ic >= 0) and (ic < self.parent.Nc_data): 
						Df[ic]+=D[ir,ip,it]
						Ndf[ic]+=1
		
		
	def compute_cell_centers(self):
		self.Rc=0.5*(self.R[0:-1,0:-1,0:-1]+self.R[1:,1:,1:])
		self.zc=0.5*(self.z[0:-1,0:-1,0:-1]+self.z[1:,1:,1:])
		self.phic=0.5*(self.phi[0:-1]+self.phi[1:])
	
	def compute_volume_and_area(self,force_computation=False):
		if (self.volumepath==self.parent.gridpath) and not force_computation:return
		def areatriangle(u,v):
			return 0.5*abs(u[0]*v[1]-u[1]*v[0])
		
		Nr=self.Nir-1;Np=self.Nip-1;Nt=self.Nit-1
		Nit=self.Nit
		self.total_volume=0
		self.volume=np.zeros((Nr,Np,Nt))
		dphi=np.zeros(Nt)
		area=self.area=np.zeros((Nr,Np,Nit))
		R=np.zeros((Nr,Np,Nit))
		Rz=np.zeros((self.Nir,self.Nip,2))
		for it in range(Nit):
			Rz[:,:,0]=self.R[:,:,it]
			Rz[:,:,1]=self.z[:,:,it]
			for ir0 in range(Nr):
				ir1=ir0+1
				for ip0 in range(Np):
					ip1=ip0+1
					area[ir0,ip0,it]=areatriangle(Rz[ir0,ip1,:]-Rz[ir0,ip0,:],Rz[ir1,ip0,:]-Rz[ir0,ip0,:])+\
					                 areatriangle(Rz[ir0,ip1,:]-Rz[ir1,ip1,:],Rz[ir1,ip0,:]-Rz[ir1,ip1,:])
					R[ir0,ip0,it]=0.25*(self.R[ir0,ip0,it]+self.R[ir1,ip0,it]+self.R[ir1,ip1,it]+self.R[ir0,ip1,it])

		for it0 in range(Nt):
			it1=it0+1
			dphi=(abs(self.phi[it1]-self.phi[it0]) % 360.)*pi/180.
			self.volume[:,:,it0]=0.5*(R[:,:,it0]+R[:,:,it1])*dphi*0.5*(area[:,:,it0]+area[:,:,it1])
		
		self.total_volume=np.sum(self.volume)
		self.volumepath=self.parent.gridpath

		
class read_target_profile():
	def __init__(self,f,sep=None):		
		#convention: 
		#R(ip,it) contains the R-coordinate of the cell edges
		#Rc(ip,it) contains the R-coordinate of the cell center
		#Rpol(ip) contains the R-coordinate of the cell edges if tor.symmetric
		#Rpolc(ip) contains the R-coordinate of the cell center if tor.symmetric
		#the s coordinate is the poloidal arc length starting from the first poloidal cell edge
		#if the target is toroidally symmetric and a curve 'sep' is given, the s=0 position is 
		# placed at the intersection point (the strike point)
		
		
		s=f.readline()
		self.name=s.strip()
		totvals=np.fromstring(f.readline(),dtype=float,count=2,sep=' ')
		self.jtot=totvals[0];self.Ptot=totvals[1]
		nn=np.fromfile(f,dtype=int,count=2,sep=' ')
		Np=nn[0];Nip=self.Nip=Np+1
		Nt=nn[1];Nit=self.Nit=Nt+1
		R=self.R=np.fromfile(f,dtype=float,count=Nip*Nit,sep=' ').reshape(Nit,Nip).T*0.01
		z=self.z=np.fromfile(f,dtype=float,count=Nip*Nit,sep=' ').reshape(Nit,Nip).T*0.01
		phi=self.phi=np.fromfile(f,dtype=float,count=Nip*Nit,sep=' ').reshape(Nit,Nip).T
		
		Rc   =self.Rc =0.5*(R[0:-1,0:-1]  +R[1:,1:])
		zc   =self.zc =0.5*(z[0:-1,0:-1]  +z[1:,1:])
		phic=self.phic=0.5*(phi[0:-1,0:-1]+phi[1:,1:])

		toroidalalign=True
		for it in range(Nit):
			toroidalalign=toroidalalign and (np.std(phi[:,it])<1e-6)
		if not toroidalalign:
			print "poloidal direction of the target is not the same as for the machine."
			return

		self.j =np.fromfile(f,dtype=float,count=Np*Nt,sep=' ').reshape(Nt,Np).T*1e4  # conversion from A/cm^2 to A/m^2
		self.P =np.fromfile(f,dtype=float,count=Np*Nt,sep=' ').reshape(Nt,Np).T*1e4  # conversion from W/cm^2 to W/m^2
		self.ne=np.fromfile(f,dtype=float,count=Np*Nt,sep=' ').reshape(Nt,Np).T*1e6  # conversion from 1/cm^3 to 1/m^3
		self.Te=np.fromfile(f,dtype=float,count=Np*Nt,sep=' ').reshape(Nt,Np).T # eV
		self.Ti=np.fromfile(f,dtype=float,count=Np*Nt,sep=' ').reshape(Nt,Np).T # eV
			
		field=self.field={}
		field['j']=self.j;field['P']=self.P;field['ne']=self.ne;field['Te']=self.Te;field['Ti']=self.Ti
		self.fieldnames=['j','P','ne','Te','Ti','jpol','Ppol','nepol','Tepol','Tipol']
		
		dA=self.dA=np.ones((Np,Nt));nperp=self.nperp=np.zeros((Np,Nt,3))
		xyz=self.xyz=np.zeros((Nip,Nit,3));xyz[:,:,0]=R*cos(phi);xyz[:,:,1]=R*sin(phi);xyz[:,:,2]=z
		
		for it in range(Nt):
			for ip in range(Np):
				dA[ip,it]=0.5*np.linalg.norm(cross(xyz[ip+1,it,:]-xyz[ip,it,:]    ,xyz[ip,it+1,:]-xyz[ip,it,:]))+\
					  0.5*np.linalg.norm(cross(xyz[ip+1,it,:]-xyz[ip+1,it+1,:],xyz[ip,it+1,:]-xyz[ip+1,it+1,:]))
				nperp[ip,it,:]=cross(xyz[ip+1,it,:]-xyz[ip,it,:],xyz[ip,it+1,:]-xyz[ip,it,:])
				nperp[ip,it,:]/=np.linalg.norm(nperp[ip,it,:])
		A=self.A=np.sum(dA)

		
		self.jpol=np.zeros(Np);self.Ppol=np.zeros(Np);self.nepol=np.zeros(Np);self.Tepol=np.zeros(Np);self.Tipol=np.zeros(Np)
		self.toroidalsym=True
		for ip in range(Np):
			Ap=np.sum(self.dA[ip,:])
			self.jpol[ip]=np.sum(self.j[ip,:]*dA[ip,:])/Ap;field['jpol']=self.jpol
			self.Ppol[ip]=np.sum(self.P[ip,:]*dA[ip,:])/Ap;field['Ppol']=self.Ppol
			self.nepol[ip]=np.sum(self.ne[ip,:]*dA[ip,:])/Ap;field['nepol']=self.nepol
			self.Tepol[ip]=np.sum(self.Te[ip,:]*dA[ip,:])/Ap;field['Tepol']=self.Tepol
			self.Tipol[ip]=np.sum(self.Ti[ip,:]*dA[ip,:])/Ap;field['Tipol']=self.Tipol
			self.toroidalsym=self.toroidalsym and \
			                 (np.std(self.Rc[ip,:])<1e-6) and (np.std(self.zc[ip,:])<1e-6)
			
		if self.toroidalsym:			
			self.Rpol=self.R[:,0];self.zpol=self.z[:,0]
			self.Rzpol=np.vstack((self.Rpol,self.zpol)).T
			self.Rpolc=self.Rc[:,0];self.zpolc=self.zc[:,0]
			self.Rzpolc=np.vstack((self.Rpolc,self.zpolc)).T
		
		try:	
			self.scoordinate(sep=sep)
		except:
			print 'error determining scoordinate'
		

	def scoordinate(self,sep=None):	
		s=self.s=np.zeros((self.Nip,self.Nit));xyz=self.xyz
		for it in range(self.Nit):
			for ip in range(self.Nip-1):
				s[ip+1,it]=s[ip,it]+norm(xyz[ip+1,it,:]-xyz[ip,it,:])
		sc=self.sc=0.5*(s[0:-1,0:-1]+s[1:,1:])
		if self.toroidalsym:			
			ssp=0.0
			if sep !=None:			
				ssp=self.ssp=curve_curve_col(self.Rzpol,sep)[0]
				self.sp=clen2x(self.Rzpol,ssp)				
			self.s-=ssp;self.sc-=ssp

	def s2Rz(self,s):
		return clen2x(self.Rzpol,s+self.ssp)	
		
	
	def smooth_gauss(self,ssm=5e-3,which=['jpol','Ppol','nepol','Tepol','Tipol']):
		dsc=np.diff(self.sc[:,0])
		if np.min(dsc)<=0.0:raise Exception('zero length sc interval or wrong order.')
		Ns=int(np.sum(dsc)/np.min(dsc))
		slin=np.linspace(np.min(self.sc[:,0]),np.max(self.sc[:,0]),Ns)
		Nh=max([int(ssm/np.min(dsc)),1])
		print "Ns,Nh=",Ns,Nh
		
		g=np.exp(-np.linspace(-2,2,2*Nh+1)**2);g=g/np.sum(g)
		
		for nam in which:
		   data=np.interp(slin,self.sc[:,0],self.field[nam]) # map to linear interval
		   ii=np.where(~np.isnan(data))[0]
		   ii=ii[np.where( (ii<=(data.shape[0]-Nh)) * (ii>Nh))]
		   new=data*0.0;count=data*0.0		
		   for i in range(-Nh,Nh):
			new[ii+i]+=data[ii]*g[i+Nh]
			count[ii+i]+=g[i+Nh]
			
		   self.field[nam+'smooth']=np.interp(self.sc[:,0],slin,new/count) #map back
		   
		
			
		
		
class read_curve():
	def __init__(self,fn,scaling_factor=0.01):
		f=open(fn,'r')
		self.Np=np.fromstring(f.readline(),dtype=int,count=1,sep=' ')[0]

		self.Rz=np.fromfile(f,dtype=float,count=2*self.Np,sep=' ').reshape(self.Np,2)*scaling_factor
		self.R=self.Rz[:,0];self.z=self.Rz[:,1]

		f.close()
		
#class read_vessel_geo():
#	def __init__(self,fn):
#		f=open(fn,'r')
#		Nc=self.Ncomponents=np.fromstring(f.readline(),dtype=int,count=1,sep=' ')[0]
#		self.components=[];self.names=[];self.validity=[];self.filled=[]
#		for ic in range(Nc):
#			self.names.append(f.readline())
#			nn=np.fromfile(f,dtype=int,count=4,sep=' ')
#			self.validity.append(nn[2]);self.filled.append(nn[3])
#			Rz=np.fromfile(f,dtype=float,count=2*nn[1],sep=' ').reshape(nn[1],2)
#			self.components.append(Rz)
#		f.close()

class read_vessel_geo():
	def __init__(self,fn):
		f=open(fn,'r')
		Nc=np.fromstring(f.readline(),dtype=int,count=1,sep=' ')[0]
		self.components=[]
		for ic in range(Nc):
			name=f.readline().strip()
			nn=np.fromfile(f,dtype=int,count=4,sep=' ')
			Rz=np.fromfile(f,dtype=float,count=2*nn[1],sep=' ').reshape(nn[1],2)
			self.components.append({'name':name,'valid':nn[2],'filled':nn[3],'Rz':Rz})
		f.close()
		
class read_limiter_geo():
	def __init__(self,fn,comp_cartesian=False,scaling_factor=0.01):
		f=open(fn,'r')
		self.comment=f.readline()
		s=f.readline()
		Nit,Nip=self.Nit,self.Nip=np.fromstring(s,count=2,sep=' ',dtype=int)
		Rz=self.Rz=np.zeros((Nip,Nit,2))
		phi=self.phi=np.zeros(Nit)
		for it in range(Nit):
			phi[it]=np.fromfile(f,count=1,sep=' ',dtype=float)[0]
			Rz[:,it,:]=np.fromfile(f,count=Nip*2,sep=' ',dtype=float).reshape(Nip,2)*scaling_factor
		f.close()
		self.R=self.Rz[:,:,0];self.z=self.Rz[:,:,1]
				
		if comp_cartesian:self.compute_cartesian()
        
	def compute_cartesian(self):
		self.xyz=np.zeros((self.Nip,self.Nit,3))
		x=self.x=self.xyz[:,:,0]
		y=self.y=self.xyz[:,:,1]
		self.xyz[:,:,2]=self.z
		for it in range(self.Nit):
			x[:,it]=self.R[:,it]*np.cos(self.phi[it]*np.pi/180.)
			y[:,it]=self.R[:,it]*np.sin(self.phi[it]*np.pi/180.)				

	def compute_area(self):
		Np=self.Nip-1;Nt=self.Nit-1
		dA=self.dA=np.ones((Np,Nt))
		xyz=self.xyz		
		for it in range(Nt):
			for ip in range(Np):
				dA[ip,it]=0.5*np.linalg.norm(cross(xyz[ip+1,it,:]-xyz[ip,it,:]    ,xyz[ip,it+1,:]-xyz[ip,it,:]))+\
					  0.5*np.linalg.norm(cross(xyz[ip+1,it,:]-xyz[ip+1,it+1,:],xyz[ip,it+1,:]-xyz[ip+1,it+1,:]))

	
	def as_triangles(self,closepol=False,closetor=True):
		if closepol and closetor:
			raise Exception('you should not close the limiter in both directions.')
		Nip=self.Nip;Nit=self.Nit
		Np=Nip if closepol else (Nip-1)
		Nt=Nit if closetor else (Nit-1)
		xyz=self.xyz
		tr=self.triangles=np.zeros((Np*Nt*2,3,3))
		print xyz.shape
		print tr.shape
		i=0
		for ip in range(Np):
			for it in range(Nt):
				tr[i,0,:]=xyz[ ip    % Nip, it    % Nit,:]
				tr[i,1,:]=xyz[(ip+1) % Nip, it    % Nit,:]
				tr[i,2,:]=xyz[(ip+1) % Nip,(it+1) % Nit,:]
				i=i+1
				tr[i,0,:]=xyz[ ip    % Nip, it    % Nit,:]
				tr[i,1,:]=xyz[ ip    % Nip,(it+1) % Nit,:]
				tr[i,2,:]=xyz[(ip+1) % Nip,(it+1) % Nit,:]
				i=i+1        
				
		return tr		
		
		
	
	def tor_symmetric(self):
		return np.all([np.all(self.R[:,it]==self.R[:,0]) for it in range(1,self.Nit)]) and\
  		       np.all([np.all(self.z[:,it]==self.z[:,0]) for it in range(1,self.Nit)])
		
		
	def writeout(self,fn,Rshift=0.0,zshift=0.0,phishift=0.0,comment=None):
		f=open(fn,'w')
		f.write(comment if comment!=None else fn)
		f.write('\n%8i %8i 0   0.0  0.0\n' % (self.Nit,self.Nip))
		for it in range(self.Nit):
			f.write('%13.4f\n' % (self.phi[it]+phishift))
			for ip in range(self.Nip):
				f.write('%13.3f%13.3f\n' % ((self.Rz[ip,it,0]+Rshift)*100.,(self.Rz[ip,it,1]+zshift)*100.))
		
		f.close()
	
class write_torsym_target_geo():
	def __init__(self,fn,c,phi1,phi2,N,comment=None,scaling_factor=100.0):
		if comment==None:comment=fn
		f=open(fn,'w')
		f.write(comment+'\n')
		f.write('%i  %i   1      0.00000      0.00000\n'% (N,c.shape[0]))
		for phi in np.linspace(phi1,phi2,N):
			f.write("%15.5f\n" % phi)
			for cc in c*scaling_factor:f.write("%15.5f   %15.5f\n" % (cc[0],cc[1]))
		f.close()
		
#-----------------------------------------------------------------------------------------
#-----------------    some service functions ---------------------------------------------
#-----------------------------------------------------------------------------------------

def completefilename(fn):
	print 'fn=',fn
	if '*' in fn:
		l=glob.glob(fn)
		if len(l)==1:
			return l[0]
		else:
			print "non-unique file name: ",l
	else:
		return fn


def searchpath(sp):
	for f in sp.split(';'):	
		ff=completefilename(f)
		if os.path.isfile(ff):return ff
	return completefilename(sp)
		

def M2v(M):return np.squeeze(np.array(M))#formally converts a matrix into a vector

def line_line_col(a0,a1,b0,b1):
	ss=np.array([1.0,1.0])
	M=np.matrix((a1-a0,b0-b1))
	if ((M[0,0]*M[1,1]-M[0,1]*M[1,0]) != 0):
		s=M2v((b0-a0)*inv(M))
		if (s[0] >= 0.0) and (s[0] < 1.0) and (s[1] >= 0.0) and (s[1] < 1.0): ss=s
	return ss

def curve_curve_col(c1,c2,real_space=False):
	n1=c1.shape[0]
	n2=c2.shape[0]
	l1=0.0 
	for i1 in range(n1-1):
		l2=0.0
		for i2 in range(n2-1):

			c=line_line_col(c1[i1,:],c1[i1+1,:],c2[i2,:],c2[i2+1,:])
			l2+=np.linalg.norm(c2[i2+1,:]-c2[i2,:])*c[1]
			if (c[0] != 1.0) or (c[1] != 1.0): 
				if real_space:
					return c1[i1,:]+(c1[i1+1,:]-c1[i1,:])*c[0]
				break  
 
		l1+=np.linalg.norm(c1[i1+1,:]-c1[i1,:])*c[0]
		#; print,c1[*,i1]+(c1[*,i1+1]-c1[*,i1])*c[0]
		if (c[0] != 1.0) or (c[1] != 1.0): break  
	return l1,l2

def clen2x(c,l):
	cl=l
	res=[-1.0,-1.0]
	for i in range(c.shape[0]-1):
		dl=np.linalg.norm(c[i+1,:]-c[i,:])
		if cl > dl :
			cl=cl-dl 
		else :
			return c[i,:]+(c[i+1,:]-c[i,:])*cl/dl
	return res

def vectorlist(a,b,N=200):
	a=np.array(a)
	b=np.array(b)	
	return [a+(b-a)*x for x in np.linspace(0.0,1.0,N)]
