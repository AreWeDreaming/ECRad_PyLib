import numpy as np
from plotting_configuration import *
import sys
import os
sys.path.append('/afs/ipp/u/tal/tools/python_plot_routines/')
from emc3_reader import *

r = emc3_reader()
r.read_grid(fn=os.path.join("/afs/ipp/u/sdenk/Documentation/Data" , 'grid.txt'), toiz=3)

col = ['red', 'blue', 'green']

plt.close('all')

fig, ax = plt.subplots(figsize=(8, 12))

for iz, zn in enumerate(r.zone):
	it0 = (zn.Nit - 1) / 2
	for ir in range(zn.Nir):
		ax.plot(zn.R[ir, :, it0], zn.z[ir, :, it0], "+", color=col[iz])
	for ip in range(zn.Nip):
		ax.plot(zn.R[:, ip, it0], zn.z[:, ip, it0], "+", color=col[iz])

ax.set_xlabel('R [m]')
ax.set_ylabel('z [m]')
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('grid.pdf')
plt.show()
