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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from imgtok.model import CosmoGridImageAutoEncoder, CosmoGridImageAutoEncoderAdv  # noqa: E402
from plot_plot import plot_astronomy_comparison  # noqa: E402
from ssim_psnr_iou import evaluate_metrics  # noqa: E402


def _load_model(ckpt_path: str, prefer_adv: bool) -> torch.nn.Module:
    if prefer_adv:
        try:
            model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
            model.eval()
            return model
        except Exception:
            model = CosmoGridImageAutoEncoder.load_from_checkpoint(ckpt_path, map_location="cpu")
            model.eval()
            return model
    try:
        model = CosmoGridImageAutoEncoder.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        return model
    except Exception:
        model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        return model


def _preprocess(sample: np.ndarray, log_mean: float, log_stddev: float) -> torch.Tensor:
    if sample.ndim == 2:
        sample = sample[np.newaxis, np.newaxis, :, :]
    elif sample.ndim == 3:
        sample = sample[np.newaxis, :, :, :]
    x = np.clip(sample, 1e-10, None)
    x = np.log(x)
    x = (x - log_mean) / log_stddev
    return torch.from_numpy(x).float()


def _postprocess(reconstructed: torch.Tensor, log_mean: float, log_stddev: float) -> np.ndarray:
    reconstructed_np = reconstructed.detach().cpu().numpy()
    reconstructed_np = reconstructed_np * log_stddev + log_mean
    reconstructed_np = np.exp(reconstructed_np)
    return reconstructed_np.squeeze()


def _parse_shape(shape_str: str) -> tuple[int, ...] | None:
    s = (shape_str or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _to_2d_for_plot(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    return np.squeeze(arr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/si0009256k6u/default/Tomasz-FHNW/CosmoGrid_Working/vqvae_update_paper/cosmogrid_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--prefer-adv", action="store_true")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--samples-per-segment", type=int, default=0)
    parser.add_argument("--psnr-threshold", type=float, default=35.0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--expected-shape", type=str, default="192,128,128")
    parser.add_argument("--best-npy-name", type=str, default="")
    parser.add_argument("--plot-threshold", action="store_true")
    parser.add_argument("--no-plot-best", action="store_false", dest="plot_best", default=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    index_path = data_root / f"{args.split}-index.json"
    content_dir = data_root / "content"

    if not args.ckpt:
        args.ckpt = (
            "/mnt/si0009256k6u/default/Tomasz-FHNW/CosmoGrid_Working/vqvae_update_paper/checkpoints/"
            "epoch=0999-step=024000-val_mse=0.145829.ckpt"
        )

    model = _load_model(args.ckpt, prefer_adv=args.prefer_adv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    log_mean = -4.9179
    log_stddev = 0.4947

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_shape = _parse_shape(args.expected_shape)
    if not args.best_npy_name:
        if expected_shape:
            shape_tag = "x".join(str(x) for x in expected_shape)
            args.best_npy_name = f"{args.split}_best_psnr_{shape_tag}.npy"
        else:
            args.best_npy_name = f"{args.split}_best_psnr.npy"

    total_segments = len(index)
    max_segments = total_segments if args.max_segments <= 0 else min(total_segments, args.max_segments)
    print(f"Split={args.split} segments={total_segments} processing={max_segments}")

    best_psnr = -float("inf")
    best_original: np.ndarray | None = None
    best_pred: np.ndarray | None = None
    best_meta: tuple[int, int, str] | None = None
    scanned = 0
    matched_shape = 0

    for seg_idx, (start_idx, end_idx, filepath) in enumerate(index[:max_segments]):
        data_path = content_dir / filepath
        data = np.load(data_path, mmap_mode="r")
        seg_len = int(end_idx - start_idx)

        if args.samples_per_segment <= 0:
            num_samples = min(seg_len, len(data))
        else:
            num_samples = min(args.samples_per_segment, seg_len, len(data))
        for local_i in range(num_samples):
            sample = np.asarray(data[local_i])
            original = sample.copy()
            scanned += 1
            if expected_shape is not None and tuple(original.shape) != expected_shape:
                continue
            matched_shape += 1

            x = _preprocess(sample, log_mean, log_stddev).to(device)
            with torch.no_grad():
                reconstructed = model.model(x)
            pred = _postprocess(reconstructed, log_mean, log_stddev)

            data_range = float(original.max() - original.min())
            psnr, ssim, iou = evaluate_metrics(pred, original, data_range=data_range)
            if args.plot_threshold and psnr > args.psnr_threshold:
                print(f"{args.split} seg={seg_idx+1}/{max_segments} file={filepath}")
                print(f"PSNR={psnr:.4f} SSIM={ssim:.6f} IoU={iou:.6f}")
                plot_astronomy_comparison(original, pred, psnr)

            if psnr > best_psnr:
                best_psnr = float(psnr)
                best_original = original
                best_pred = pred
                best_meta = (seg_idx, local_i, filepath)

    if best_original is None or best_pred is None or best_meta is None:
        raise RuntimeError(
            f"No sample found for expected-shape={expected_shape}. scanned={scanned} matched_shape={matched_shape}"
        )

    best_path = output_dir / args.best_npy_name
    np.save(best_path, best_original)
    seg_idx, local_i, filepath = best_meta
    print(
        f"Best PSNR={best_psnr:.6f} saved={best_path} seg={seg_idx+1}/{max_segments} i={local_i} file={filepath}"
    )
    if args.plot_best:
        plot_astronomy_comparison(_to_2d_for_plot(best_original), _to_2d_for_plot(best_pred), best_psnr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
