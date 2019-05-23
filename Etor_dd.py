#!/usr/bin/env python

import numpy as np
from plotting_configuration import *
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
from matplotlib import rc
# from IPython import embed
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.path.append("/afs/ipp-garching.mpg.de/aug/ads-diags/common/python/lib")
sys.path.append("../ECRad_Pylib")
# sys.path.append('../../../misc/')
# import myplotlib as my
# plt.ion()
from scipy.signal import kaiserord, lfilter, firwin
from colorsys import hls_to_rgb
# rc('text', usetex=True)

'''
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N
rm = running_mean
'''

def fir(t, data, ripple_db=30, width=15, cutoff_hz=4):
    # adapted from http://wiki.scipy.org/Cookbook/FIRFilter
    if len(t) < 2:
        return t, data
    sample_rate = 1. / (t[1] - t[0])
    nyq_rate = sample_rate / 2.0
    # ripple_db, width, cutoff_hz = args.fir
    width = width / nyq_rate
    N, beta = kaiserord(ripple_db, width)
    try:
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))
    except ValueError, e:
        print "nyq:", nyq_rate
        raise e
    delay = 0.5 * (N - 1) / sample_rate

    if len(data.shape) == 2:
        return t[N - 1:] - delay, np.array([lfilter(taps, 1.0, data[:, i])[N - 1:] for i in xrange(data.shape[1])]).T
    else:
        return t[N - 1:] - delay, lfilter(taps, 1.0, data)[N - 1:]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rm(x, a):
    return np.average(rolling_window(x, a), -1)

def rs(x, a):
    return np.std(rolling_window(x, a), -1)

# shot = 32303 # 32298
# shot = 32298
# shot = 32342

def getUprofile(shot, exp='ABOCK', eqi='IDE', ed=0, t=None, t_smooth=None):

    filename = '/ptmp1/work/sdenk/pfls/%i_%s_%s_%i.npz' % (shot, exp, eqi, ed)
    print filename
    # embed()
    if os.path.exists(filename):
        data = np.load(filename)
        pfls = data['pfls']
        psisAx = data['psisAx']
        psisSep = data['psisSep']
        ts = data['ts']
        rts = data['rts']
    else:
        import dd
        import kk_abock as kk
        eqsf = dd.shotfile(eqi, shot, exp)
        ts = eqsf('time').data

        eq = kk.kk()
        eq.Open(shot, exp, eqi)

        N = 81 if eqi == 'IDE' else 41
        # N = 82 if eqi == 'IDE' and shot == 33134 else N

        arrN = N if shot != 33134 else N + 1
        pfls = np.zeros((len(ts), arrN))
        psisAx = []
        psisSep = []
        rts = np.zeros((len(ts), arrN))
        for i, t in enumerate(ts):
            print i, t, ts[-1], shot, exp, eqi, ed
            pfls[i] = eq.get_jpar(t, N)['pfl']
            psiAx, psiSep = eq.get_special_points(t)['pfxx'][:2]
            psisAx.append(psiAx)
            psisSep.append(psiSep)
            rts[i] = eq.rhopol_to_rhotor(t, np.sqrt((pfls[i] - psiAx) / (psiSep - psiAx)))['rhotor']

        psisAx = np.array(psisAx)
        psisSep = np.array(psisSep)
        np.savez(filename, psisAx=psisAx, psisSep=psisSep, pfls=pfls, ts=ts, rts=rts)

    # rhos = np.sqrt((pfls.T-psisAx)/(psisSep-psisAx)).T

    rhos = (rts[1:] + rts[:-1]) / 2.  # (rhos[1:] + rhos[:-1])/2.
    Us = -((pfls[1:] - pfls[:-1]).T / (ts[1:] - ts[:-1])).T
    nt = (ts[1:] + ts[:-1]) / 2.
    if(t is None):
        t_range = np.where(ts > 1)
        s = 0.1
    else:
        t_range = np.where(np.logical_and(ts > t - t_smooth, ts < t + t_smooth))
        s = 0.05
    ts_psi_mat = ts[t_range]
    rhot_mean = np.mean(rts[t_range], axis=0)[::-1]
    # Interpolated spline for the mapping
    Psi_mat_spl = RectBivariateSpline(ts[t_range], rhot_mean, pfls[:, ::-1][t_range], s=0)
    smoothed_mat = []
    for i in range(len(rhot_mean)):
        t_filter, psi_smooth = fir(ts_psi_mat, Psi_mat_spl(ts_psi_mat, rhot_mean[i], grid=False))
        smoothed_mat.append(psi_smooth)
    Psi_mat_spl = RectBivariateSpline(t_filter, rhot_mean, np.array(smoothed_mat).T, s=s)
    return nt, rhos, Us, ts, psisAx, psisSep, pfls, ts_psi_mat, rhot_mean, Psi_mat_spl

def makeTrace(nt, Us, i, n=20):
    return rm(nt, n), rm(Us[:, i], n), rs(Us[:, i], n)


if __name__ == '__main__':
    colors = []  # my.colors(8)
    n = 12
    for i in range(n):
        colors.append(hls_to_rgb(float(i) / n, 0.5, 0.5))
    # my.PGF()
    fig1 = plt.figure(111)
    res = {} if 'res' not in globals() else res
    cnt = -1
    # ax3d = plt.subplot(1,1,1, projection='3d') if 'ax3d' not in globals() else ax3d
    # ax3d.cla()
    # shots = [31113, 31163, 32298, 32303, 32304, 32305, 32342]
    shots = [33697]
    # shots = [32342]
    # colors = ['purple', 'red', 'blue', 'black']
    plt.clf()
    for shot, color in zip(shots, colors):
        # if shot not in (32298, 32303, 32304, 32305):
        #    continue
        print shot
        cnt += 1
        if shot in res:
            nt, rhos, Us, ts, psisAx, psisSep, pfls, ts_psi_mat, rhot_mean, Psi_mat_spl = res[shot]
        else:
            nt, rhos, Us, ts, psisAx, psisSep, pfls, ts_psi_mat, rhot_mean, Psi_mat_spl = getUprofile(shot, exp="AUGD")  # if shot != 32232 else getUprofile(shot, 'AUGD', 'EQH')
            # , t=1.68, t_smooth=0.1
            res[shot] = [nt, rhos, Us, ts, psisAx, psisSep, pfls]
            # embed()
            lastshot = shot
        print(ts.shape, pfls.shape, rhos.shape)
        ax1 = fig1.add_subplot(2, 1, cnt + 1)
        ax1.cla()
        tmp, = ax1.plot(ts, psisAx, "+", label=r"$\Psi_\mathrm{ax}$")
        ax1.plot(ts, psisSep, "*", label=r"$\Psi_\mathrm{sep}$", color=tmp.get_color())
        ft, fd = fir(ts, psisAx)
        psi_Ax_spl = UnivariateSpline(ft, fd, s=1.e-5)
        ax1.plot(ft, psi_Ax_spl(ft), color='red', lw=2, label=r"$\Psi_\mathrm{ax} \mathrm{(smooth)}$")
        ft, fd = fir(ts, psisSep)
        psi_sep_spl = UnivariateSpline(ft, fd, s=1.e-5)
        ax1.plot(ft, psi_sep_spl(ft), color=colors[0], lw=2, label=r"$\Psi_\mathrm{sep} \mathrm{(smooth)}$")
 #        ax1.plot(ts, rhos.T[0], color=colors[1], lw=2, label=r"$\rho_\mathrm{tor\,ax}$")
 #        ax1.plot(ts, rhos.T[-1], color=colors[2], lw=2, label=r"$\rho_\mathrm{tor\,sep}$")
        ax1.set_ylabel(r"$\Psi_\mathrm{pol}\,\mathrm{[Vs]}$")
        ax2 = fig1.add_subplot(2, 1, cnt + 2, sharex=ax1)
        ax2.plot(ft, psi_Ax_spl(ft, 1), label=r"$U_\mathrm{loop,\,ax}$")
        ax2.plot(ft, psi_sep_spl(ft, 1), label=r"$U_\mathrm{loop,\,sep}$")
        ax2.set_xlabel(r"$t\,\mathrm{[s]}$")
        ax2.set_ylabel(r"$U_\mathrm{loop}\,\mathrm{[V]}$")
        ax1.legend(loc="best")
        ax2.legend(loc="best")
 #        for i in range(len(ts[ts > 1][::20])):
 #            fig2 = plt.figure(211)
 #            ax3 = fig2.add_subplot(1, 1, 1)
 #            ax3.plot(rhot_mean, Psi_mat_spl(ts[ts > 1][::20][i], rhot_mean, grid=False))
 #            plt.show()
        fig2 = plt.figure(211)
        ax3 = fig2.add_subplot(1, 1, 1)
        V_loop_mat = Psi_mat_spl(ts_psi_mat, rhot_mean, dx=1).T
        for i in range(len(rhot_mean[::10])):
            ax3.plot(ts_psi_mat, V_loop_mat[::10][i], color=colors[i], label=r"$V_\mathrm{loop}\,\rho_\mathrm{tor} = " + r"{0:1.2f}$".format(rhot_mean[::10][i]))
        ax3.set_xlabel(r"$t\,[\mathrm{s}]$")
        ax3.set_ylabel(r"$V_\mathrm{loop}\,[\mathrm{V}]$")
        ax3.legend()
        fig3 = plt.figure(311)
        ax4 = fig3.add_subplot(1, 1, 1)
        levels = np.linspace(np.min(V_loop_mat.flatten()), np.max(V_loop_mat.flatten()), 50)
        cmap = plt.cm.get_cmap("plasma")
        cont1 = ax4.contourf(ts_psi_mat, rhot_mean, V_loop_mat, levels=levels, cmap=cmap)
        cont2 = ax4.contour(ts_psi_mat, rhot_mean, V_loop_mat, levels=levels, colors='k', \
                            hold='on', alpha=0.25, linewidths=1)
        ax4.set_xlabel(r"$t\,[\mathrm{s}]$")

        ax4.set_ylabel(r"$\rho_\mathrm{tor}$")
        for c in cont2.collections:
            c.set_linestyle('solid')
        cb = fig2.colorbar(cont1, ax=ax4, ticks=np.linspace(np.round(np.min(levels), 1), np.round(np.max(levels), 1), 5))  #
        cb.set_label(r"$V_\mathrm{loop}\,[\mathrm{V}]$")
        plt.show()
        # plt.subplot(2, 2, cnt + 1)
#        if shot != 32303:
#            stride = 1
#            plt.plot(ts[::stride], pfls[::stride, ::10], color=color, lw=1)
#            ft, fd = fir(ts, pfls)
#            # plt.plot(ft, fd[:, ::10], color='red', lw=1.5)
#            n = 20
#            # plt.plot(rm(ft, n), -rm(np.gradient(fd, 1e-3, 1e-3)[0].T, n).T[:,[0,-1]])
#        else:
#            mseind = ts < 3.7
#            nomseind = 3.7 <= ts
#            plt.plot(ts[mseind], pfls[mseind, ::10], color=color, lw=1)
#            plt.plot(ts[nomseind], pfls[nomseind, ::10], color=color, lw=1, alpha=0.4)
#
#
#        # plt.grid(True)
#
#
#
#        plt.ylim(-0.2, 1.5)
#        plt.xlim(0.1, 5.5)
#        plt.xticks(range(1, 6))
#        if shot < 32304:
#            plt.gca().set_xticklabels([])
#        else:
#            plt.xlabel(r'$\mathrm{time~[s]}$')
#        if shot % 2 != 0:
#            plt.gca().set_yticklabels([])
#
#
#        # plt.ylabel(r'$\psi($axis .. separatrix$)$ [Vs]')
#        plt.title(str(shot), loc='right', y=0.835, x=0.97, fontdict={'fontsize':10},
#            bbox={'facecolor':'white', 'edgecolor':'white', 'pad':0})
#
#        pass
#
#    plt.gcf().text(0.01, 0.55, r'$\psi(\mathrm{axis .. separatrix})~\mathrm{[Vs]}$', va='center', rotation='vertical')
#    plt.tight_layout()
#    plt.show()
    # plt.savefig('Etor.pdf')
#    try:
#        __IPYTHON__
#    except:
#        embed()
#        pass











