# -*- coding: utf-8 -*-
"""
CosmoGrid run_adv — patch-level reconstruction comparison plot.

Loads one fully-loaded CosmoGrid field [192,128,128], reconstructs every patch,
selects a few representative patches (the ones with highest PSNR), and saves a
side-by-side "original | reconstruction" mosaic to assets/patch_examples.png.

Usage:
    python plot_patch.py \
        --ckpt weights/best_cosmogrid_adv.ckpt \
        --content-file data/cosmogrid_data/content/baryonified512_036.npy \
        --out-png assets/patch_examples.png \
        --n-patches 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from imgtok.model import CosmoGridImageAutoEncoderAdv  # noqa: E402

# ---- normalization constants (must match training) ---------------------------
LOG_MEAN = -4.9179
LOG_STD = 0.4947


def preprocess(sample: np.ndarray) -> torch.Tensor:
    if sample.ndim == 2:
        sample = sample[np.newaxis, np.newaxis, :, :]
    x = np.clip(sample, 1e-10, None)
    x = np.log(x)
    x = (x - LOG_MEAN) / LOG_STD
    return torch.from_numpy(x).float()


def postprocess(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy() * LOG_STD + LOG_MEAN
    return np.exp(x)


def load_checkpoint(ckpt_path: str):
    model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def psnr(pred: np.ndarray, orig: np.ndarray) -> float:
    mse = float(np.mean((pred - orig) ** 2))
    if mse < 1e-10:
        return float("inf")
    dr = float(orig.max() - orig.min())
    return 20.0 * np.log10(dr + 1e-12) - 10.0 * np.log10(mse + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run_adv checkpoint")
    ap.add_argument("--content-file", required=True, help="one field [192,128,128] .npy")
    ap.add_argument("--out-png", default="assets/patch_examples.png")
    ap.add_argument("--n-patches", type=int, default=12)
    args = ap.parse_args()

    if args.n_patches > 192:
        raise ValueError("n-patches cannot exceed 192")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt).to(device)

    field = np.load(args.content_file)  # (192,128,128)
    if field.ndim != 3:
        raise ValueError(f"expected [192,128,128], got {field.shape}")
    print(f"Loaded field {args.content_file}: {field.shape}")

    # reconstruct every patch
    recon = np.empty_like(field)
    for i in range(field.shape[0]):
        x = preprocess(field[i]).to(device)
        with torch.no_grad():
            out = model.model(x)
        recon[i] = postprocess(out).squeeze()

    # choose the `n` highest-PSNR patches so examples are visually faithful
    scores = [psnr(recon[i], field[i]) for i in range(field.shape[0])]
    chosen = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.n_patches]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = args.n_patches
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 4.4))
    if ncols == 1:
        axes = axes[:, None]

    # Log-scale the values once and derive a shared color range via percentiles
    # so the colormap spans the actually-plotted range with good contrast (fixes
    # the previous dark/clipped image, which used raw min/max on log data).
    log_field = np.log10(field + 1e-10)
    log_recon = np.log10(recon + 1e-10)
    log_all = np.concatenate([log_field.ravel(), log_recon.ravel()])
    vmin = float(np.percentile(log_all, 2.0))
    vmax = float(np.percentile(log_all, 98.0))

    for j, idx in enumerate(chosen):
        axo, axr = axes[0, j], axes[1, j]
        axo.imshow(log_field[idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axo.set_title(f"original {idx}", fontsize=9)
        axo.axis("off")
        axr.imshow(log_recon[idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axr.set_title(f"adv-recon (PSNR {scores[idx]:.1f})", fontsize=9)
        axr.axis("off")

    plt.suptitle("run_adv — original (top) | adv-reconstruction (bottom)", fontsize=12)
    plt.tight_layout()

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved patch comparison -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())