# -*- coding: utf-8 -*-
"""
CosmoGrid run_adv — end-to-end inference demo.

Loads the run_adv FSQ autoencoder (CosmoGridImageAutoEncoderAdv), reconstructs the
192 equatorial patches of one CosmoGrid field, composes them into a full nside=512
HEALPix sky map (12 x 512 x 512), and optionally:

  - saves the composed sky maps (original / reconstruction) as .npy
  - computes the TT angular power spectrum and saves a comparison figure
  - saves a montage of reconstructed patches

Usage:
    python inference.py \
        --ckpt weights/best_cosmogrid_adv.ckpt \
        --data-root data/cosmogrid_data \
        --content-file data/cosmogrid_data/content/baryonified512_036.npy \
        --rearr rearr_nside512.npy \
        --out-dir ./outputs \
        --save-maps --save-cl --save-patches
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from imgtok.model import CosmoGridImageAutoEncoderAdv  # noqa: E402


# ---- normalization constants (match training) --------------------------------
LOG_MEAN = -4.9179
LOG_STD = 0.4947


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    """Load the run_adv checkpoint (hyperparameters are stored inside)."""
    model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model.to(device)


def preprocess(sample: np.ndarray) -> torch.Tensor:
    """(H, W) float32 dark-matter field -> normalized tensor (1,1,H,W)."""
    if sample.ndim == 2:
        sample = sample[np.newaxis, np.newaxis, :, :]
    x = np.clip(sample, 1e-10, None)
    x = np.log(x)
    x = (x - LOG_MEAN) / LOG_STD
    return torch.from_numpy(x).float()


def postprocess(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy() * LOG_STD + LOG_MEAN
    return np.exp(x)


def run_patch_inference(model, patches: np.ndarray, device: str) -> np.ndarray:
    """patches: (N,128,128) float32 -> recon (N,128,128)."""
    recons = []
    for i in range(patches.shape[0]):
        x = preprocess(patches[i]).to(device)
        with torch.no_grad():
            recon = model.model(x)
        recons.append(postprocess(recon).squeeze())
    return np.stack(recons, axis=0)


def compose_sky_map(patch_volume: np.ndarray, rearr: np.ndarray) -> np.ndarray:
    """
    patch_volume: (192,128,128) float32 (the reconstructed field).
    rearr: length-3145728 int array; rearr[h] = flat grid index of HEALPix pixel h.

    Returns flat nside=512 map (3145728,) ordered by HEALPix pixel index,
    which can be reshaped to (12,512,512) for the nested ordering.
    """
    flat = patch_volume.reshape(-1)  # 3145728
    return flat[rearr]


def save_patch_montage(orig: np.ndarray, recon: np.ndarray, out_png: str, n_cols: int = 12):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = orig.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    vmin = float(min(orig.min(), recon.min()))
    vmax = float(max(orig.max(), recon.max()))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(recon[i], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"patch {i}", fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--content-file", required=True)
    ap.add_argument("--rearr", default="rearr_nside512.npy")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--save-cl", action="store_true")
    ap.add_argument("--save-patches", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    # 1. load one full field = 192 patches
    content = Path(args.data_root) / "content" / Path(args.content_file).name
    data = np.load(content, mmap_mode="r")  # (192,128,128)
    field_orig = np.asarray(data)
    print(f"Loaded field: {content} shape={field_orig.shape} dtype={field_orig.dtype}")

    # 2. patch-level reconstruction (192 patches)
    recon = run_patch_inference(model, field_orig, device)
    print(f"Reconstructed {recon.shape[0]} patches in [{recon.shape[1]}x{recon.shape[2]}]")

    # 3. compose into full nside=512 sky map
    rearr = np.load(args.rearr).astype(np.int64)
    orig_sky = compose_sky_map(field_orig, rearr)
    recon_sky = compose_sky_map(recon, rearr)
    nside = 512
    npix = nside * nside * 12
    assert orig_sky.shape == (npix,), f"expected {npix}, got {orig_sky.shape}"

    # also keep the (12,512,512) nested view for inspection
    nested_orig = orig_sky.reshape(12, nside, nside)
    nested_recon = recon_sky.reshape(12, nside, nside)
    print(f"Composed sky maps: flat {orig_sky.shape}, nested {nested_orig.shape}")

    if args.save_maps:
        np.save(out_dir / "sky_orig_flat.npy", orig_sky)
        np.save(out_dir / "sky_recon_flat.npy", recon_sky)
        np.save(out_dir / "sky_orig_nested.npy", nested_orig)
        np.save(out_dir / "sky_recon_nested.npy", nested_recon)
        print("Saved sky maps (flat + nested) to", out_dir)

    if args.save_patches:
        png = out_dir / "recon_patch_montage.png"
        save_patch_montage(field_orig, recon, str(png))
        print("Saved patch montage ->", png)

    if args.save_cl:
        import healpy as hp

        lmax = 1000
        cl_orig = hp.anafast(orig_sky, lmax=lmax)
        cl_recon = hp.anafast(recon_sky, lmax=lmax)
        ls = np.arange(len(cl_orig))
        np.save(out_dir / "cl_orig.npy", cl_orig)
        np.save(out_dir / "cl_recon.npy", cl_recon)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"hspace": 0.08, "height_ratios": [2, 1]},
        )
        ax1.loglog(ls[2:], cl_orig[2:], c="black", lw=1.5, label=r"$C_\ell^{\rm orig}$")
        ax1.loglog(ls[2:], cl_recon[2:], c="#d62728", ls="--", lw=1.2, label=r"$C_\ell^{\rm recon}$")
        ax1.legend()
        ax1.set_ylabel(r"$C_\ell$")
        ax1.set_xlim(2, lmax)
        ratio = cl_recon[2:] / cl_orig[2:]
        ax2.semilogx(ls[2:], ratio, c="#d62728", lw=1.0)
        ax2.axhline(1.0, c="gray", ls="--", lw=0.8)
        ax2.set_ylabel("ratio")
        ax2.set_xlabel(r"Multipole $\ell$")
        fig.tight_layout()
        fig.savefig(out_dir / "cl_comparison.png", bbox_inches="tight", dpi=180)
        plt.close(fig)
        print("Saved C_l comparison ->", out_dir / "cl_comparison.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())