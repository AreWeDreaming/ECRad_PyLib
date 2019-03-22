import numpy as np
import dd
# import IPython


class EQU:

    def __init__(self , Shotnumber=None):
        self.Status = False
        if Shotnumber is not None :
            self.Load(Shotnumber)

    def __del__(self):
        self.Unload()
        del self.Status

    def Load(self , Shotnumber, Experiment='AUGD', Diagnostic='EQI', Edition=0L):

        self.Unload()
        if Diagnostic == 'EQI' or Diagnostic == 'EQH' or Diagnostic == 'EQB'  or Diagnostic == 'IDE':
            try:
                if(Diagnostic == 'EQB'):
                    try:
                        sf = dd.shotfile(Diagnostic, Shotnumber, Experiment, Edition, diagnostic2="EQK")
                    except Exception as e:
                        print("Getting EQB with EQK name failed trying EQI")
                        sf = dd.shotfile(Diagnostic, Shotnumber, Experiment, Edition, diagnostic2="EQI")
                else:
                    sf = dd.shotfile(Diagnostic, Shotnumber, Experiment, Edition)
                self.Shotnumber = Shotnumber
            except Exception as e:
                print "Error reading shotfile"
                print("Error:", e)
                return False

            self.Nz = sf.getParameter('PARMV', 'N').data + 1
            self.NR = sf.getParameter('PARMV', 'M').data + 1
            self.Ntime = sf.getParameter('PARMV', 'NTIME').data + 1
            self.R = (sf.getSignalGroup("Ri"))[0:self.Ntime, 0:self.NR]
            self.z = (sf.getSignalGroup("Zj"))[0:self.Ntime, 0:self.Nz]
            self.time = (sf.getSignal("time"))[0:self.Ntime]
            self.PsiOrigin = sf.getSignalGroup("PFM")[0:self.Ntime, 0:self.Nz, 0:self.NR]
            # #time, R, z
            self.Psi = np.swapaxes(self.PsiOrigin, 1, 2)
            self.ed = sf.edition
            sf.close()

            self.Status = True
        return True


    def Unload(self):
        if self.Status:
            self.Status = False
            del self.Nz
            del self.NR
            del self.Ntime
            del self.R
            del self.z
            del self.time
            del self.Psi


    def getPsi(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.Psi[idx]

    def getR(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.R[idx]

    def getz(self, timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.z[idx]

    def __call__(self , timepoint):
        if self.Status:
            idx = np.argmin(np.abs(self.time - timepoint))
            return self.R[idx], self.z[idx], self.Psi[idx]
