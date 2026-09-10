# -*- coding: utf-8 -*-
"""
Save validation reconstruction results (origs.npy / preds.npy).

Loads the run_adv FSQ autoencoder, reconstructs all 192 patches of one
CosmoGrid field, and saves the original and reconstructed patch volumes to
<output-dir>/origs.npy and <output-dir>/preds.npy. These arrays feed the
downstream power-spectrum / patch-comparison analysis scripts.

Usage:
    python analysis/plot_cl.py \
        --ckpt weights/best_cosmogrid_adv.ckpt \
        --data-root data/cosmogrid_data \
        --content-file data/cosmogrid_data/content/baryonified512_036.npy \
        --output-dir results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent          # .../upload_github/analysis
_REPO = _ROOT.parent                              # .../upload_github
sys.path.insert(0, str(_ROOT))                    # sibling utils: plot_plot, ssim_psnr_iou
sys.path.insert(0, str(_REPO / "src"))            # imgtok, astro_utils
sys.path.insert(0, str(_REPO))                    # aion

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from imgtok.model import CosmoGridImageAutoEncoder, CosmoGridImageAutoEncoderAdv  # noqa: E402
from ssim_psnr_iou import *  # noqa: E402,F401,F403
from plot_plot import *  # noqa: E402,F401,F403

# 与训练一致的归一化常数
LOG_MEAN = -4.9179
LOG_STDDEV = 0.4947


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    try:
        model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
    except Exception:
        model = CosmoGridImageAutoEncoder.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(_REPO / "weights" / "best_cosmogrid_adv.ckpt"))
    ap.add_argument("--data-root", default=str(_REPO / "data" / "cosmogrid_data"))
    ap.add_argument("--content-file", default=str(_REPO / "data" / "cosmogrid_data" / "content" / "baryonified512_036.npy"))
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    # 读取索引 (仅用于报告规模)
    index_path = Path(args.data_root) / f"{args.split}-index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    print(f"Found {len(index)} {args.split} segments in index")

    # 加载一个完整场 (192 个 patch)
    data_path = Path(args.content_file)
    data = np.load(data_path, mmap_mode="r")
    print(f"Loaded {data_path}: shape={data.shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origs, preds = [], []
    for local_i in range(data.shape[0]):
        sample = np.asarray(data[local_i])
        original_sample = sample.copy()

        # (H, W) -> (1, 1, H, W)
        if sample.ndim == 2:
            sample = sample[np.newaxis, np.newaxis, :, :]
        else:
            sample = sample[np.newaxis, :, :, :]

        # 预处理: log + 标准化
        x = np.clip(sample, 1e-10, None)
        x = np.log(x)
        x = (x - LOG_MEAN) / LOG_STDDEV
        x = torch.from_numpy(x).float().to(device)

        # 推理
        with torch.no_grad():
            reconstructed = model.model(x)
            reconstructed_np = reconstructed.cpu().numpy()
            reconstructed_np = reconstructed_np * LOG_STDDEV + LOG_MEAN
            reconstructed_np = np.exp(reconstructed_np)

            origs.append(original_sample)
            preds.append(reconstructed_np.squeeze())

    origs = np.array(origs)
    preds = np.array(preds)
    np.save(output_dir / "origs.npy", origs)
    np.save(output_dir / "preds.npy", preds)
    print(f"Saved origs {origs.shape} and preds {preds.shape} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
