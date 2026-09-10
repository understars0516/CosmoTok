import os
import pylab as pl
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


rearr = np.load("rearr_nside512.npy").astype("int")

orig_adv_map = np.load("orig_adv.npy").reshape(-1)[rearr]
orig_base_map = np.load("orig_base.npy").reshape(-1)[rearr]

pred_adv_map = np.load("pred_adv.npy").reshape(-1)[rearr]
pred_base_map = np.load("pred_base.npy").reshape(-1)[rearr]

orig_base = hp.anafast(orig_base_map, lmax=2000)
orig_adv = hp.anafast(orig_adv_map, lmax=2000)

pred_base = hp.anafast(pred_base_map, lmax=2000)
pred_adv = hp.anafast(pred_adv_map, lmax=2000)

ls = np.arange(len(pred_base))

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, width_ratios=[1], height_ratios=[1.5, 1])
ax1, ax2 = gs.subplots(sharex='col', sharey='row')
ax1.loglog(ls, orig_base, label = r"$C_\ell^{Orig}$", c='black')
ax1.loglog(ls, pred_base, label = r"$C_\ell^{Pred_{base}}$", c='blue', linestyle='--')
ax1.loglog(ls, pred_adv, label = r"$C_\ell^{Pred_{adv}}$", c='red', linestyle='--')

ax1.set_xlim(1.5, 2000)
ax1.set_xticks([2, 10, 100, 500, 2000])
ax1.set_xticklabels(['2', '10', '100', '500', '2000'])
ax1.set_ylim(1e-13, 1e-7)
ax1.set_yticks([1e-13, 1e-11, 1e-9, 1e-7])
ax1.set_yticklabels([r'$10^{-13}$', r'$10^{-11}$', r'$10^{-9}$',  r'$10^{-7}$'])
ax1.set_ylabel(r"$C^{XX}_\ell$")
ax1.legend(loc='best', ncol=1)


ax_top = ax1.twiny()
ax_top.spines.top.set_position(("axes", 1))  # 调整上部axis的位置
ax_top.set_xscale("log")
ax_top.set_xlim(1.5, 2000)
ax_top.set_xticks([2, 10, 100, 500, 2000])
ax_top.set_xticklabels([r'$90^\circ$', r'$18^\circ$', r'$1.8^\circ$', r'$0.36^\circ$', r'$0.09^\circ$'])
ax_top.set_xlabel("Angular scale")
ax_top.spines['left'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['right'].set_visible(False)

ax2.loglog(ls, pred_base / orig_base, label = r"Pred_base / Orig_base", color='lightblue', linestyle='--')
ax2.loglog(ls, pred_adv / orig_adv, label = r"Pred_adv / Orig_adv", color='pink', linestyle='--')
ax2.loglog(ls, ls/ls, color='gray', label=r'$Ratio=1$', linestyle='--')

ax2.minorticks_off()
ax2.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")
ax2.set_ylabel(r"$Ratio$")
#ax2.set_ylim(1e-4, 1e3)
#ax2.set_yticks([1e-4, 1e-2, 1e0, 1e2])
#ax2.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',  r'$10^{0}$', r'$10^{2}$'])
ax2.set_xlim(1.5, 2000)
ax2.set_xticks([2, 10, 100, 500, 2000])
ax2.set_xticklabels(['2', '10', '100', '500', '2000'])
ax2.set_ylabel(r"$Ratio$")
ax2.legend(ncol=1, loc='lower left')


for ax in fig.get_axes():
    ax.label_outer()
plt.savefig("cls_ratio.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

