# -*- coding: utf-8 -*-
"""
Three-way patch comparison plot: original vs base-recon vs adv-recon.

Draws a 2x3 dashboard for a single patch:
    orig | pred_base | pred_adv | |pred_base/orig - 1| | |pred_adv/orig - 1| | pred_base/pred_adv

Requires three intermediate patch volumes in <results-dir>:
    orig.npy, pred_base.npy, pred_adv.npy   (each shaped [192,128,128])
Generate them with plot_cl.py (base + adv checkpoints) and rename the saved
origs.npy / preds.npy accordingly:

    python analysis/plot_cl.py --ckpt weights/best_cosmogrid_base.ckpt --output-dir results
    mv results/origs.npy results/orig.npy
    mv results/preds.npy results/pred_base.npy
    # repeat with the adv checkpoint -> pred_adv.npy

Usage:
    python analysis/plot_patch.py --results-dir results --sample-index 100 \
        --output comparison_dashboard.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

_ROOT = Path(__file__).resolve().parent          # .../upload_github/analysis
_REPO = _ROOT.parent                              # .../upload_github

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def plot_prediction_comparison(orig, pred1, pred2, save_path=None, dpi=300, show=False):
    """Draw the 2x3 comparison dashboard for a single patch.

    Parameters
    ----------
    orig, pred1, pred2 : np.ndarray
        Ground-truth patch and the two reconstructions (H, W).
    save_path : str or None
        Output PNG path (None -> only return the figure).
    dpi : int
    show : bool
        If True, call plt.show() (interactive use only).
    """
    eps = 1e-8  # 防止除零

    # 相对误差
    err1 = (pred1 / (orig + eps) - 1.0) / 10
    err2 = (pred2 / (orig + eps) - 1.0) / 10

    # pred1/pred2 比值
    ratio = pred1 / (pred2 + eps) - 1

    # 统一原图与预测图的显示范围
    vmin_data = min(orig.min(), pred1.min(), pred2.min())
    vmax_data = max(orig.max(), pred1.min(), pred2.min())

    # 误差图上限(clip掉极端异常值)
    err_vmax = max(np.percentile(err1, 99), np.percentile(err2, 99))
    if err_vmax <= 0:
        err_vmax = 1e-6

    # 比值图对称范围(以1为中心)
    ratio_center = 1.0
    ratio_dev = max(abs(ratio.min() - 1.0), abs(ratio.max() - 1.0))
    ratio_dev = min(ratio_dev, 2.0)  # clip极端比值
    ratio_vmin = ratio_center - ratio_dev
    ratio_vmax = ratio_center + ratio_dev

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    panel_configs = [
        (r'$Orig$',                    orig,   'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{base}$',             pred1,  'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{adv}$',              pred2,  'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{base}/Orig - 1$',    err1,   'YlOrRd',   0,           err_vmax),
        (r'$Pred_{adv}/Orig - 1$',     err2,   'YlOrRd',   0,           err_vmax),
        (r'$Pred_{base} / Pred_{adv}$', ratio, 'RdBu_r',   ratio_vmin,  ratio_vmax),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes.flat, panel_configs):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=22, pad=12)
        ax.axis('off')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _require(results_dir: Path, name: str) -> np.ndarray:
    p = results_dir / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing intermediate file {p}. Generate it with plot_cl.py first "
            f"(see the module docstring):\n"
            f"  python analysis/plot_cl.py --ckpt weights/best_cosmogrid_adv.ckpt "
            f"--output-dir {results_dir}"
        )
    return np.load(p, mmap_mode="r")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="directory containing orig.npy / pred_base.npy / pred_adv.npy")
    ap.add_argument("--orig-name", default="orig.npy")
    ap.add_argument("--pred1-name", default="pred_base.npy")
    ap.add_argument("--pred2-name", default="pred_adv.npy")
    ap.add_argument("--sample-index", type=int, default=0,
                    help="which patch (of the 192) to plot (default: 0)")
    ap.add_argument("--output", default="comparison_dashboard.png")
    ap.add_argument("--show", action="store_true",
                    help="show the figure interactively (default: save only)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    orig = _require(results_dir, args.orig_name)
    pred1 = _require(results_dir, args.pred1_name)
    pred2 = _require(results_dir, args.pred2_name)

    idx = args.sample_index
    if idx < 0 or idx >= orig.shape[0]:
        raise IndexError(
            f"sample-index {idx} out of range for array with {orig.shape[0]} patches"
        )

    plot_prediction_comparison(
        np.asarray(orig[idx]), np.asarray(pred1[idx]), np.asarray(pred2[idx]),
        save_path=args.output, show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
