# CosmoTok

This repository packages everything needed to **run inference** with the adversarial
**run_adv** model (`CosmoGridImageAutoEncoderAdv`) on CosmoGrid cosmological
simulation maps: the trained checkpoint, the source code, the dataset layout, and the
results generated with it.

The model is a **Finite Scalar Quantizer (FSQ) + PatchGAN + LPIPS** adversarial
autoencoder: each 128×128 input patch is mapped to a discrete latent code, which is
then decoded back into a reconstructed field. A full CosmoGrid field contains **192
patches**, which together tile a **nside = 512 HEALPix sky map** (12 × 512² = 3,145,728
pixels).

---

## ⚡ Quick restore — get the model & data runnable quickly

The **checkpoint** (`weights/best_cosmogrid_adv.ckpt`, ~1.99 GB) and the
**dataset field images** (`data/cosmogrid_data/content/*.npy`, 100 files, ~1.2 GB)
are **too large for Git/GitHub**. They are hosted on **Google Drive**:

| Asset | Where it goes |
|---|---|
| `weights/best_cosmogrid_adv.ckpt` | [Google Drive → best_cosmogrid_adv.ckpt](<GOOGLE_DRIVE_WEIGHTS_LINK>) |
| `data/cosmogrid_data/content/*.npy` | [Google Drive → cosmogrid content/](<GOOGLE_DRIVE_DATA_LINK>) |

After downloading from Drive, place the files at the exact paths above
(create the folder `data/cosmogrid_data/content/` and drop the 100
`baryonified512_*.npy` files there; put the `.ckpt` under `weights/`).

Then you can run inference:

```bash
# from the repository root
python3 inference.py --ckpt weights/best_cosmogrid_adv.ckpt \
  --data-root data/cosmogrid_data \
  --content-file data/cosmogrid_data/content/baryonified512_036.npy \
  --rearr ../rearr_nside512.npy \
  --out-dir ./assets --save-maps --save-cl --save-patches
```

> The small index files (`index.json`, `train/valid/test-index.json`,
> `stats.pkl`) and the 25 MB `arr_nside512_192x128x128.npy` layout map are
> **already committed to this repo**, so only the two Google-Drive downloads
> above are needed.

---

## Repository layout

```
.
├── README.md                     ← you are here
├── inference.py                  ← end-to-end inference demo (192 patches → sky map)
├── run_adv.sh                    ← launch training of the adv model (reference)
├── cfg-cosmo-adv.yaml            ← run_adv training config
├── cfg-cosmo-adv-embed8.yaml     ← variant config (embedding dim 8)
├── src/
│   ├── imgtok/                   ← main package (model, data, train, common)
│   ├── astro_utils/              ← helper deps used by imgtok
│   └── imgemb/                   ← embedding/inference helpers
├── aion/                         ← 'aion' codec library (aion.codecs.*) source dep
├── data/
│   └── cosmogrid_data/           ← dataset (indexes + stats committed; content/*.npy on Google Drive)
├── weights/
│   └── best_cosmogrid_adv.ckpt   ← checkpoint — NOT in git, download from Google Drive
└── assets/                       ← generated analysis images (this README embeds them)
```

## Dependencies

- `torch`, `numpy`, `scipy`, `einops`, `jaxtyping`, `jsonargparse`
- `lightning.pytorch` (≥ 2.x), `torchmetrics`
- `healpy`, `matplotlib` (for inference plotting)
- optional `wandb` (only if you re-run training)
- optional `lpips` (used only at training time for perceptual loss; inference falls
  back gracefully)

> The `aion` folder is a source dependency (a local Python package), **not** the
> `polymathic-aion` published package. Keep it on `PYTHONPATH` (see `inference.py`).

---

## Quick start: run inference

> **First time?** The checkpoint and the dataset field files are NOT in this git repo.
> Download them from the Google Drive links in the **Quick restore** section above and
> place them at `weights/best_cosmogrid_adv.ckpt` and
> `data/cosmogrid_data/content/*.npy`.

Make the checkpoint, data indexes and the `rearr` permutation visible, then run:

```bash
# from the repo root
python inference.py \
  --ckpt weights/best_cosmogrid_adv.ckpt \
  --data-root data/cosmogrid_data \
  --content-file data/cosmogrid_data/content/baryonified512_036.npy \
  --rearr ../rearr_nside512.npy \
  --out-dir ./assets \
  --save-maps --save-cl --save-patches
```

What it does:

1. Loads the run_adv checkpoint (`CosmoGridImageAutoEncoderAdv`, hyper-parameters are
   stored inside the checkpoint).
2. Loads **one full field** — a `(192, 128, 128)` array, i.e. 192 patches of 128×128.
3. Runs patch-level inference → produces 192 reconstructed patches.
4. Uses `rearr_nside512.npy` (a permutation of length 3,145,728) to **compose the 192
   patches into one nside=512 HEALPix map**:

   ```python
   sky_flat  = recon_patches.reshape(-1)[rearr]        # (3145728,)
   sky_nested = sky_flat.reshape(12, 512, 512)          # HEALPix nested view
   ```

5. Optionally saves flat/nested sky maps, the angular TT power spectrum
   (`cl_orig.npy`, `cl_recon.npy`), and a patch example figure.

### Outputs

| File | Description |
|---|---|
| `assets/sky_orig_flat.npy` | original field composed into flat nside=512 map |
| `assets/sky_recon_flat.npy` | run_adv reconstruction as flat nside=512 map |
| `assets/sky_orig_nested.npy` | same, reshaped `(12, 512, 512)` nested view |
| `assets/sky_recon_nested.npy` | same, reshaped `(12, 512, 512)` nested view |
| `assets/cl_orig.npy` / `cl_recon.npy` | TT angular power spectra |
| `assets/cl_comparison.png` | power-spectrum comparison figure |
| `assets/patch_examples.png` | original vs reconstruction patch pairings |

---

## Generated images (run_adv results)

> All images below live in `assets/` and are produced by `inference.py` / the analysis
> scripts included in this pack.

### Composed nside=512 sky map (original vs reconstruction)

![Composed original vs run_adv reconstruction](assets/sky_mollview.png)

The 192 patches are stacked and rearranged with `rearr_nside512.npy` into a full
HEALPix nside=512 sky map. **Top:** original field. **Bottom:** run_adv reconstruction.

### Sample 128×128 patches (original vs adv reconstruction)

![Patch-level reconstruction examples (PSNR ≥ 40 dB)](assets/patch_examples.png)

Example patches (original | adv-reconstruction) selected specifically to show cases with
**PSNR ≥ 40 dB** (40.06–41.70 dB). The adversarial (LPIPS + PatchGAN) objective keeps
small-scale filamentary structure crisp compared to a pure-MSE model.

### TT angular power spectrum

![Angular power spectrum comparison](assets/cl_comparison.png)

`C_ℓ` of the composed original map vs the composed run_adv reconstruction, with the
ratio in the bottom panel. The reconstruction preserves the power spectrum down to
arc-minute scales (up to ℓ ≈ 1000 with the current field).

> To reproduce fresh figures, replace the `content-file` argument with any field in
> `data/cosmogrid_data/content/`.

---

## Analysis code

The `analysis/` folder contains the scripts used to produce the figures:

- `inference.py` — unified pipeline (192 patches → sky map → C_ℓ → figures).
- `plot.py` — classic plotting of original vs reconstruction with metrics.
- `plot_cl.py`, `plot_cl_hp2.py`, `plot_cls.py` — power-spectrum (`C_ℓ`) analyses; they
  use `healpy.anafast`, and expect `rearr_nside512.npy` for composition.
- `plot_patch.py`, `plot_plot.py` — patch-grid and comparison dashboards.
- `ssim_psnr_iou.py` — NumPy-only implementation of SSIM/PSNR/IoU metrics used in the
  scripts.

The `rearr_nside512.npy` permutation is referenced by the analysis scripts; it is
provided alongside this pack (keep it next to the data or update the path).

---

## Inference analysis: composing an HEALPix sky map & computing the power spectrum (step by step)

> These steps run fully offline. The only inputs needed are
> `weights/best_cosmogrid_adv.ckpt`, the content files `data/cosmogrid_data/content/*.npy`,
> and `rearr_nside512.npy`.

### Step 1 — Run inference over the 192 patches

A full sky field (one `baryonified512_XXX.npy` file) is a `(192, 128, 128)` tensor,
i.e. **192 patches of 128×128**. The model works at patch level, so inference is done
patch by patch:

```python
import numpy as np, torch
from imgtok.model import CosmoGridImageAutoEncoderAdv

# Normalization constants (must match training)
LOG_MEAN, LOG_STD = -4.9179, 0.4947

model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(
    "weights/best_cosmogrid_adv.ckpt", map_location="cpu").eval().cuda()

field = np.load("data/cosmogrid_data/content/baryonified512_036.npy")  # (192,128,128)
recons_patches = np.zeros_like(field)                                 # stores 192 recon patches

for i in range(field.shape[0]):
    patch = field[i]                                    # (128,128) float32
    x = np.clip(patch, 1e-10, None)
    x = np.log(x)                                       # log transform
    x = (x - LOG_MEAN) / LOG_STD                        # standardize -> (1,1,128,128)
    x = torch.from_numpy(x[np.newaxis]).float().cuda()
    with torch.no_grad():
        recon = model.model(x)                          # forward pass (includes FSQ quantization)
    recon = recon.cpu().numpy() * LOG_STD + LOG_MEAN    # de-standardize
    recons_patches[i] = np.exp(recon).squeeze()         # exponentiate back to brightness units
```

`recons_patches` now has shape `(192, 128, 128)`, one reconstructed patch per input field patch.

### Step 2 — Compose the 192 patches into one HEALPix sky map

The data is the **nside = 512 HEALPix sky map stored as tiles**:
`12 × 512 × 512 = 3,145,728 = 192 × 128 × 128` pixels.

`rearr_nside512.npy` is an index array of length 3,145,728:
`rearr[h]` gives "which linear position of the flattened 192×128×128 grid HEALPix
pixel h lives at".

Composition formula:

```python
rearr = np.load("rearr_nside512.npy").astype(np.int64)   # (3145728,)

# Flatten the 192 reconstructed patches, then re-order by HEALPix pixel index = full sky map
sky_flat = recons_patches.reshape(-1)[rearr]             # (3145728,)

# (12, 512, 512) view ordered as HEALPix nested (12 HEALPix base pixels)
sky_nested = sky_flat.reshape(12, 512, 512)
```

- `sky_flat` is the complete all-sky map in **HEALPix pixel-index order**;
- `sky_nested` is its `(12, 512, 512)` view for visualization and downstream `healpy` use;
- the `orig` map is built the same way: `orig.reshape(-1)[rearr]`.

> In short, `rearr_nside512.npy` is the "patch coordinates → HEALPix pixel" mapping;
> without it the field cannot be reassembled into a sky map.

### Step 3 — Power-spectrum analysis with `healpy`

Once the full sky map is available, use standard HEALPix tools to compute the
anisotropic angular power spectrum:

```python
import healpy as hp

lmax = 1000                      # maximum multipole
cl_orig  = hp.anafast(sky_orig_flat,  lmax=lmax)   # C_ℓ of original map
cl_recon = hp.anafast(sky_recon_flat, lmax=lmax)   # C_ℓ of reconstructed map

# D_ℓ = ℓ(ℓ+1)/2π · C_ℓ, keeping only ℓ ≥ 2
ell = np.arange(len(cl_orig))
D_l_orig  = ell * (ell + 1) / (2 * np.pi) * cl_orig
D_l_recon = ell * (ell + 1) / (2 * np.pi) * cl_recon
```

**Power-spectrum difference / ratio** (reconstruction fidelity metrics):

```python
ratio  = D_l_recon[2:] / D_l_orig[2:]           # per-ℓ ratio, ideal value 1.0
delta  = D_l_recon[2:] - D_l_orig[2:]           # absolute difference, same units as D_ℓ
# relative deviation in dB:
rel_dev_dB = 10 * np.log10(ratio)               # dB (0 dB = perfect match)

# Summary statistics
print("median ratio :", np.median(ratio))
print("|Δ| max      :", np.abs(delta).max())
print("median |dB|  :", np.median(np.abs(rel_dev_dB)))
```

### Step 4 — Plot the power-spectrum comparison (optional)

`analysis/plot_cl.py` and `analysis/plot_cl_hp2.py` produce two stacked panels:
the top panel shows `D_ℓ` (log scale) original vs reconstructed, the bottom panel shows
the per-ℓ ratio or difference, so you can see at a glance which scales are preserved
and which deviate.

The `assets/` folder already contains power-spectrum figures generated by these scripts:

**`assets/cl_comparison.png`** — two stacked panels:
top: `D_ℓ` (log scale) original (black) vs `run_adv` reconstruction (red dashed);
bottom: per-ℓ `D_ℓ` difference (ideal = flat line at 0):

![TT power-spectrum comparison (top: D_ℓ, bottom: difference)](assets/cl_comparison.png)

**`assets/cls_ratio.png`** — `C_ℓ` with ratio, base/adv joint comparison:
bottom panel is the `Pred/Orig` ratio, `Ratio = 1` means perfect fidelity:

![C_ℓ ratio (base/adv comparison)](assets/cls_ratio.png)

> To regenerate the power-spectrum figures for **any other field**, just point
> `content-file` at a different file under `data/cosmogrid_data/content/`; the scripts above
> automatically do 192-patch inference → sky-map composition → `healpy.anafast` → plotting.

---

## Data layout

`data/cosmogrid_data/`:
- `content/baryonified512_XXX.npy` — each is a full field: shape `(192, 128, 128)`
  float32 (192 patches).
- `index.json` — global segment index `[start, end, rel_path]`.
- `train-index.json`, `valid-index.json`, `test-index.json` — split indexes used by
  `SPDLCosmogridDataset` (`src/imgtok/data/datasets.py`).
- `stats.pkl` — dataset statistics (mean/std used for log-normal preprocessing).

> If you move the data, update `root_dir` in `cfg-cosmo-adv*.yaml` and the script
> arguments accordingly.

## Training (reference, optional)

The checkpoints in this pack were produced with the `run_adv` recipe:

```bash
# one GPU; override device count if needed
CUDA_VISIBLE_DEVICES=8 bash run_adv.sh
```

See `cfg-cosmo-adv.yaml` for all model/data/trainer settings (FSQ levels `[17,17,17,17,13,13,13,13]`,
PatchGAN `NLayerDiscriminator`, hinge loss, LPIPS weight `0.01`, MSE pixel weight `1.0`,
AdamW lr `5e-5`, cosine schedule).

---

## Migration notes

1. **Large files**. `weights/best_cosmogrid_adv.ckpt` is ~1.9 GB and
   `data/cosmogrid_data/content/*.npy` total ~1.2 GB. GitHub rejects files >100 MB, so
   use **Git LFS** (or a shared blob store) for these two paths.
2. **Cross-machine paths**. The config files still contain absolute `root_dir`
   pointing to the original working tree; change them to your local data location.
3. **`rearr_nside512.npy`** must stay next to the scripts (or adapt the path argument);
   without it the patch→sky-map composition is undefined.
4. `aion/` **must** be importable (`aion.codecs.*`) for the model code; it is already on
   the `PYTHONPATH` configured by `inference.py`.