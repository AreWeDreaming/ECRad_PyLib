from matplotlib import use
import wx
use('wxAGG')
from matplotlib import pyplot as plt
from ecrad_pylib.Global_Settings import globalsettings
# plt.style.use('bmh')
plot_mode = globalsettings.plot_mode  #  "Presentation"  "Article"
if(plot_mode != "Software"):
    plt.rcParams['text.latex.preamble'] = r"\usepackage{siunitx} \sisetup{detect-all} " + \
                                          r"\usepackage{sansmath} \usepackage{amsmath} " + \
                                          r"\usepackage{amsfonts} \usepackage{amssymb} " + \
                                          r"\usepackage{braket}"
# plt.rcParams['text.latex.preamble'] = [\
#       r'\usepackage{siunitx}',  \
#       r'\sisetup{detect-all}',   \
#       r'\usepackage{sansmath}', \
#       r'\usepackage{amsmath}' , \
#       r'\usepackage{amsfonts}', \
#       r'\usepackage{amssymb}', \
#       r'\usepackage{braket}', \
#       r'\sisetup{detect-all}', \
#       r"\usepackage[cm]{sfmath}"]
# r'\boldmath',\
# r'\usepackage{bm}', \
# font = {'size'   : 18}
#
if(plot_mode == "Article"):
    plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'size' : 30})  # 24
    plt.rcParams["legend.fontsize"] = 18 # 18 #20
    plt.rcParams['axes.titlesize'] = 24  # 24 #36
    plt.rcParams['axes.labelsize'] = 24  # 24 #36
    plt.rc('text', usetex=True)
elif(plot_mode == "Presentation"):
    plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'size' : 30})  # 24
    plt.rcParams["legend.fontsize"] = 18  # 18 #20
    plt.rcParams['axes.titlesize'] = 24  #  # 24 #36
    plt.rcParams['axes.labelsize'] = 24  # 24 #36
    plt.rc('text', usetex=True)
elif(plot_mode == "Software"):
    plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Computer Modern Sans serif'], 'size' : 14})  # 24
    plt.rcParams["legend.fontsize"] = 18  # 14  # 18 #20
    plt.rcParams['axes.titlesize'] = 24  # 14  # 24 #36
    plt.rcParams['axes.labelsize'] = 24  # 14  # 24 #36
    plt.rc('text', usetex=False)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.rc('font',  size='16', family = 'sans-serif')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.major.size'] = 8  # major tick size in points
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 8  # major tick size in points
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['lines.markeredgewidth'] = 2.5
plt.rcParams['axes.linewidth'] = 3
plt.rcParams["savefig.transparent"] = True
default_x1 = 12.0
default_y1 = 8.5
default_x2 = 12.0
default_y2 = 8.5
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator as NLocator
from matplotlib.patches import Circle as pltCircle
import matplotlib as mpl    
