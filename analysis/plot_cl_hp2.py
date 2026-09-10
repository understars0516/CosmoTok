# -*- coding: utf-8 -*-
"""
TT angular power spectrum comparison: original vs base-reconstruction.

Reads flat HEALPix sky maps (produced by plot_cl.py or inference.py)
and plots the power spectrum with a diff panel.

Requires intermediate files in <results-dir>:
    origs.npy, preds.npy  (each shaped [192,128,128] or flat [3145728,])
Generate them by running plot_cl.py first, e.g.:

    python analysis/plot_cl.py \
        --ckpt weights/best_cosmogrid_adv.ckpt \
        --output-dir results

Usage:
    python analysis/plot_cl_hp2.py --results-dir results --output TT_cl_comparison_diff.png
"""

import argparse
import sys
from pathlib import Path

import healpy as hp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent          # .../upload_github/analysis
_REPO = _ROOT.parent                              # .../upload_github


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


def compose_sky_map(volume: np.ndarray, rearr: np.ndarray) -> np.ndarray:
    """Flatten [192,128,128] and reorder with `rearr` -> flat nside=512 map."""
    return volume.reshape(-1)[rearr]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="directory containing origs.npy / preds.npy (default: results)")
    ap.add_argument("--data-root", default=str(_REPO / "data" / "cosmogrid_data"),
                    help="CosmoGrid data root (holds arr_nside512_192x128x128.npy)")
    ap.add_argument("--rearr", default=None,
                    help="precomputed rearr_nside512.npy; if omitted, derived from data-root")
    ap.add_argument("--output", default="TT_cl_comparison_diff.png",
                    help="output figure path (default: TT_cl_comparison_diff.png)")
    ap.add_argument("--lmax", type=int, default=1000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    # --- Load intermediate files ---
    def _require(name: str) -> np.ndarray:
        p = results_dir / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing intermediate file {p}. Generate it with plot_cl.py first:\n"
                f"  python analysis/plot_cl.py --ckpt weights/best_cosmogrid_adv.ckpt "
                f"--output-dir {results_dir}"
            )
        return np.load(p)

    origs_raw = _require("origs.npy")
    preds_raw = _require("preds.npy")

    rearr = load_rearr(args.rearr, args.data_root)

    # Compose flat HEALPix maps from patch volumes
    if origs_raw.ndim == 3 and origs_raw.shape[0] == 192:
        origs = compose_sky_map(origs_raw, rearr)
        preds = compose_sky_map(preds_raw, rearr)
    else:
        # Already flat maps (apply rearr directly)
        origs = origs_raw.reshape(-1)[rearr]
        preds = preds_raw.reshape(-1)[rearr]

    # ==========================================
    # 2. Compute power spectrum & diff
    # ==========================================

    pred_cl = hp.anafast(preds, lmax=args.lmax)
    orig_cl = hp.anafast(origs, lmax=args.lmax)

    max_l_avail = len(pred_cl) - 1
    l_plot = np.arange(2, min(args.lmax, max_l_avail) + 1)

    factor = l_plot * (l_plot + 1) / (2 * np.pi)

    Dl_pred = factor * pred_cl[l_plot]
    Dl_orig = factor * orig_cl[l_plot]
    Dl_diff = Dl_pred - Dl_orig

    # ==========================================
    # 3. Plot
    # ==========================================

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={'hspace': 0.05, 'height_ratios': [2, 1]}
    )

    # --- Top: power spectrum comparison (log-log) ---
    ax1.loglog(l_plot, Dl_orig, label=r"$D_\ell^{\mathrm{orig}}$",
               color='black', linewidth=1.5, linestyle='-')
    ax1.loglog(l_plot, Dl_pred, label=r"$D_\ell^{\mathrm{pred}}$",
               color='#d62728', linewidth=1.2, linestyle='--')

    ax1.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax1.set_xlim(2, l_plot[-1])
    y_min = min(Dl_orig.min(), Dl_pred.min())
    y_max = max(Dl_orig.max(), Dl_pred.max())
    ax1.set_ylim(y_min * 0.8, y_max * 1.2)
    ax1.legend(loc='best', frameon=True, framealpha=0.9)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # --- Top twin axis: angular scale ---
    ax_top = ax1.twiny()
    ax_top.set_xscale("log")
    ax_top.set_xlim(ax1.get_xlim())
    ticks_l = [2, 10, 100, 500, l_plot[-1]]
    ticks_l = [t for t in ticks_l if t <= l_plot[-1]]
    ticks_theta = [180.0 / t for t in ticks_l]
    labels_theta = []
    for t in ticks_theta:
        if t >= 1.0:
            labels_theta.append(rf"{t:.1f}$^\circ$")
        else:
            labels_theta.append(rf"{t*60:.0f}'")

    ax_top.set_xticks(ticks_l)
    ax_top.set_xticklabels(labels_theta)
    ax_top.set_xlabel("Angular Scale")
    ax_top.tick_params(axis='x', which='both', bottom=False, top=True,
                       labelbottom=False, labeltop=True)
    ax_top.spines['top'].set_visible(True)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # --- Bottom: absolute diff (linear Y, log X) ---
    ax2.semilogx(l_plot, Dl_diff, color='#d62728', linewidth=1.2,
                 label=r"$D_\ell^{\mathrm{pred}} - D_\ell^{\mathrm{orig}}$")
    ax2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.6)

    ax2.set_ylabel(r"$\Delta$")
    ax2.set_xlabel(r"Multipole Moment $\ell$")

    diff_min, diff_max = Dl_diff.min(), Dl_diff.max()
    diff_range = diff_max - diff_min
    if diff_range == 0:
        ax2.set_ylim(-1e-10, 1e-10)
    else:
        margin = diff_range * 0.1
        ax2.set_ylim(diff_min - margin, diff_max + margin)

    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend(loc='best', frameon=True, framealpha=0.9)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Plot saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())