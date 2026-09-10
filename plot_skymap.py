# -*- coding: utf-8 -*-
"""
CosmoGrid run_adv — compose a full nside=512 HEALPix sky map and plot it.

Pipeline:
    1. Load one CosmoGrid field:  a [192,128,128] array = 192 patches of 128×128.
    2. Reconstruct each patch through the run_adv FSQ autoencoder.
    3. Stack the 192 reconstructed patches into [192,128,128], flatten to
       3,145,728 elements and reorder with rearr_nside512.npy
       (rearr[h] = flat index of HEALPix pixel h) to form a full nside=512
       HEALPix sky map  (12 × 512 × 512 = 3,145,728 pixels).
    4. Plot original vs reconstruction on the HEALPix map and save
       assets/sky_mollview.png  (top: original, bottom: reconstruction).

Usage:
    python plot_skymap.py \
        --ckpt weights/best_cosmogrid_adv.ckpt \
        --content-file data/cosmogrid_data/content/baryonified512_036.npy \
        --out-png assets/sky_mollview.png

    --rearr is optional: if not given, the patch->HEALPix permutation is derived
    from <data-root>/arr_nside512_192x128x128.npy (argsort of the flattened array).
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
    """(H, W) float32 field -> normalized tensor (1,1,H,W)."""
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


def load_rearr(rearr_path: str | None, data_root: str) -> np.ndarray:
    """Load the patch->HEALPix pixel permutation (int64, shape (3145728,)).

    Priority:
      1. an explicit rearr_nside512.npy file (when --rearr is given and present);
      2. derived from <data-root>/arr_nside512_192x128x128.npy via argsort(.reshape(-1)).
    """
    if rearr_path and Path(rearr_path).is_file():
        print(f"Loading rearrangement index from {rearr_path}")
        return np.load(rearr_path).astype(np.int64)
    arr_path = Path(data_root) / "arr_nside512_192x128x128.npy"
    if arr_path.is_file():
        print(f"Deriving rearrangement index from {arr_path} (argsort of flattened array)")
        arr = np.load(arr_path).reshape(-1)
        return np.argsort(arr).astype(np.int64)
    raise FileNotFoundError(
        "Could not find a rearrangement index. Either pass --rearr <rearr_nside512.npy>, "
        f"or place arr_nside512_192x128x128.npy under the data root ({data_root})."
    )


def reconstruct_volume(model, field: np.ndarray, device) -> np.ndarray:
    """field: [192,128,128] -> recon volume [192,128,128] (per-patch inference)."""
    recons = []
    for i in range(field.shape[0]):
        x = preprocess(field[i]).to(device)
        with torch.no_grad():
            out = model.model(x)  # includes FSQ quantization
        recons.append(postprocess(out).squeeze())
    return np.stack(recons, axis=0)


def compose_sky_map(volume: np.ndarray, rearr: np.ndarray) -> np.ndarray:
    """Flatten [192,128,128] and reorder with `rearr` -> flat nside=512 map."""
    return volume.reshape(-1)[rearr]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run_adv checkpoint")
    ap.add_argument("--content-file", required=True, help="one field [192,128,128] .npy")
    ap.add_argument("--data-root", default="data/cosmogrid_data",
                    help="cosmogrid data root (holds arr_nside512_192x128x128.npy)")
    ap.add_argument("--rearr", default=None,
                    help="patch->HEALPix permutation .npy; if omitted, derived from "
                         "<data-root>/arr_nside512_192x128x128.npy")
    ap.add_argument("--out-png", default="assets/sky_mollview.png")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt).to(device)

    field = np.load(args.content_file)  # (192,128,128)
    if field.ndim != 3:
        raise ValueError(f"expected [192,128,128], got {field.shape}")
    print(f"Loaded field {args.content_file}: {field.shape}")

    # Step 1: reconstruct all 192 patches -> [192,128,128]
    recon = reconstruct_volume(model, field, device)

    # Step 2: compose flat nside=512 HEALPix maps (original & recon)
    rearr = load_rearr(args.rearr, args.data_root)
    orig_sky = compose_sky_map(field, rearr)   # (3145728,)
    recon_sky = compose_sky_map(recon, rearr)  # (3145728,)

    if orig_sky.shape != (3_145_728,):
        raise RuntimeError(f"unexpected map shape {orig_sky.shape}")
    nside = 512
    # healpy mollview expects a flat (12*nside^2,) map in RING ordering; the
    # composed maps are already flat & RING-ordered (verified against anafast).
    orig_map = orig_sky
    recon_map = recon_sky
    print("Composed original & reconstructed nside=512 maps")

    # Step 3: plot with healpy mollview
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import healpy as hp

    vmin = args.vmin if args.vmin is not None else float(min(orig_map.min(), recon_map.min()))
    vmax = args.vmax if args.vmax is not None else float(max(orig_map.max(), recon_map.max()))

    fig = plt.figure(figsize=(12, 9))
    hp.mollview(orig_map, title="Original - nside=512", cmap="inferno",
                norm="hist", min=vmin, max=vmax, sub=(2, 1, 1), cbar=True)
    hp.mollview(recon_map, title="run_adv reconstruction - nside=512", cmap="inferno",
                norm="hist", min=vmin, max=vmax, sub=(2, 1, 2), cbar=True)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved composite sky map -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())