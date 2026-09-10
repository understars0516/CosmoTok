# CosmoTok

An adversarial FSQ (Finite Scalar Quantizer) autoencoder for **CosmoGrid** cosmological maps. Each 128×128 input patch is encoded to a discrete latent code and decoded back to a reconstructed field. A full CosmoGrid field contains **192 patches**, which tile a **nside = 512 HEALPix sky map** (12 × 512² = 3,145,728 pixels).

```
inference.py   one command: 192 patches → full sky map → Cℓ → plots
```

---

## Get started

The checkpoint (`weights/best_cosmogrid_adv.ckpt`, ≈ 2 GB) and the dataset field images (`data/cosmogrid_data/content/*.npy`, ≈ 1.2 GB) are too large for GitHub — download them from Google Drive and place them at the paths shown:

| File | Destination on disk |
|---|---|
| [weights → Google Drive](https://drive.google.com/drive/folders/1KlBOyAxkIlLz1o7xO5CvJlYnyypQ8006?usp=sharing) | `weights/best_cosmogrid_adv.ckpt` |
| [data → Google Drive](https://drive.google.com/drive/folders/1bovfVOgewyQq7K2NlM2yV7fP0QCKHBIP?usp=sharing) | `data/cosmogrid_data/content/` (100 `baryonified512_*.npy`) |

Then run inference:

```bash
python3 inference.py --ckpt weights/best_cosmogrid_adv.ckpt \
  --data-root data/cosmogrid_data \
  --content-file data/cosmogrid_data/content/baryonified512_036.npy \
  --rearr ../rearr_nside512.npy \
  --out-dir assets --save-maps --save-cl --save-patches
```

The small index files (`index.json`, `train/valid/test-index.json`, `stats.pkl`, 25 MB `arr_nside512_192x128x128.npy`) are already in the repo — no further setup needed.

---

## Results

Full Composed sky map (original vs reconstruction):

![Composed sky map](assets/sky_mollview.png)

Sample patches, original | adv-reconstruction (PSNR ≥ 40 dB shown):

![Patch examples](assets/patch_examples.png)

TT angular power spectrum `Cℓ` (original vs reconstruction, ratio in bottom panel):

![Power spectrum](assets/cl_comparison.png)

Power spectrum ratio `Pred/Orig` (base vs adv):

![Power spectrum ratio](assets/cls_ratio.png)

The reconstruction preserves small-scale structure and the power spectrum down to arc-minute scales (ℓ ≈ 1000).

---

## Repository layout

```
src/            model & data code (imgtok, astro_utils, imgemb)
aion/           'aion' codec library (aion.codecs.*) — source dependency
analysis/       plotting & metric scripts (plot_cl, plot_cls, ssim_psnr_iou …)
data/cosmogrid_data/   indexes + stats (committed); content/*.npy (Google Drive)
weights/        checkpoint (Google Drive, not in git)
assets/         generated figures
```

## Dependencies

`torch`, `numpy`, `scipy`, `einops`, `jaxtyping`, `jsonargparse`, `lightning.pytorch ≥ 2.x`, `torchmetrics`, `healpy`, `matplotlib`. Optional: `wandb`, `lpips` (training only).

> `aion/` is a local source package (not the `polymathic-aion` PyPI package). Keep it on `PYTHONPATH` — `inference.py` handles this automatically.

## Train (optional)

`CUDA_VISIBLE_DEVICES=8 bash run_adv.sh` — see `cfg-cosmo-adv.yaml` for all settings.