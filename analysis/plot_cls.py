# -*- coding: utf-8 -*-
"""
Compare TT angular power spectra: original vs base-recon vs adv-recon, plus
the Pred/Orig ratios. Produces cls_ratio.png.

Requires four intermediate patch volumes in <results-dir>:
    orig_base.npy, orig_adv.npy, pred_base.npy, pred_adv.npy
(each shaped [192,128,128]). Generate them by running plot_cl.py with the base
and adv checkpoints and renaming the saved origs.npy / preds.npy, e.g.

    python analysis/plot_cl.py --ckpt weights/best_cosmogrid_base.ckpt \
        --output-dir results   # -> results/origs.npy, results/preds.npy  (base)
    mv results/origs.npy results/orig_base.npy
    mv results/preds.npy results/pred_base.npy
    # repeat with the adv checkpoint -> orig_adv.npy, pred_adv.npy

Usage:
    python analysis/plot_cls.py --results-dir results --output cls_ratio.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import healpy as hp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parent          # .../upload_github/analysis
_REPO = _ROOT.parent                              # .../upload_github

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def load_rearr(rearr_path: str | None, data_root: str) -> np.ndarray:
    """Patch->HEALPix permutation (int, shape (3145728,))."""
    if rearr_path and Path(rearr_path).is_file():
        return np.load(rearr_path).astype("int")
    arr_path = Path(data_root) / "arr_nside512_192x128x128.npy"
    if arr_path.is_file():
        arr = np.load(arr_path).reshape(-1)
        return np.argsort(arr).astype("int")
    raise FileNotFoundError(
        "No rearrangement index found. Pass --rearr <rearr_nside512.npy> or place "
        f"arr_nside512_192x128x128.npy under the data root ({data_root})."
    )


def require(results_dir: Path, name: str) -> np.ndarray:
    p = results_dir / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing intermediate file {p}. Generate it with plot_cl.py "
            f"(see the module docstring)."
        )
    return np.load(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-root", default=str(_REPO / "data" / "cosmogrid_data"))
    ap.add_argument("--rearr", default=None)
    ap.add_argument("--output", default="cls_ratio.png")
    ap.add_argument("--lmax", type=int, default=2000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rearr = load_rearr(args.rearr, args.data_root)

    orig_adv_map = require(results_dir, "orig_adv.npy").reshape(-1)[rearr]
    orig_base_map = require(results_dir, "orig_base.npy").reshape(-1)[rearr]
    pred_adv_map = require(results_dir, "pred_adv.npy").reshape(-1)[rearr]
    pred_base_map = require(results_dir, "pred_base.npy").reshape(-1)[rearr]

    orig_base = hp.anafast(orig_base_map, lmax=args.lmax)
    orig_adv = hp.anafast(orig_adv_map, lmax=args.lmax)
    pred_base = hp.anafast(pred_base_map, lmax=args.lmax)
    pred_adv = hp.anafast(pred_adv_map, lmax=args.lmax)

    ls = np.arange(len(pred_base))

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, width_ratios=[1], height_ratios=[1.5, 1])
    ax1, ax2 = gs.subplots(sharex='col', sharey='row')
    ax1.loglog(ls, orig_base, label=r"$C_\ell^{Orig}$", c='black')
    ax1.loglog(ls, pred_base, label=r"$C_\ell^{Pred_{base}}$", c='blue', linestyle='--')
    ax1.loglog(ls, pred_adv, label=r"$C_\ell^{Pred_{adv}}$", c='red', linestyle='--')

    ax1.set_xlim(1.5, args.lmax)
    ax1.set_xticks([2, 10, 100, 500, args.lmax])
    ax1.set_xticklabels(['2', '10', '100', '500', str(args.lmax)])
    ax1.set_ylim(1e-13, 1e-7)
    ax1.set_yticks([1e-13, 1e-11, 1e-9, 1e-7])
    ax1.set_yticklabels([r'$10^{-13}$', r'$10^{-11}$', r'$10^{-9}$', r'$10^{-7}$'])
    ax1.set_ylabel(r"$C^{XX}_\ell$")
    ax1.legend(loc='best', ncol=1)

    ax_top = ax1.twiny()
    ax_top.spines.top.set_position(("axes", 1))
    ax_top.set_xscale("log")
    ax_top.set_xlim(1.5, args.lmax)
    ax_top.set_xticks([2, 10, 100, 500, args.lmax])
    ax_top.set_xticklabels([r'$90^\circ$', r'$18^\circ$', r'$1.8^\circ$', r'$0.36^\circ$', r'$0.09^\circ$'])
    ax_top.set_xlabel("Angular scale")
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    ax2.loglog(ls, pred_base / orig_base, label=r"Pred_base / Orig_base", color='lightblue', linestyle='--')
    ax2.loglog(ls, pred_adv / orig_adv, label=r"Pred_adv / Orig_adv", color='pink', linestyle='--')
    ax2.axhline(1.0, color='gray', label=r'$Ratio=1$', linestyle='--')

    ax2.minorticks_off()
    ax2.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")
    ax2.set_xlim(1.5, args.lmax)
    ax2.set_xticks([2, 10, 100, 500, args.lmax])
    ax2.set_xticklabels(['2', '10', '100', '500', str(args.lmax)])
    ax2.set_ylabel(r"$Ratio$")
    ax2.legend(ncol=1, loc='lower left')

    for ax in fig.get_axes():
        ax.label_outer()
    out = Path(args.output)
    plt.savefig(out, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
