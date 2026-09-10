# save_validation_results.py
import torch
import numpy as np
from pathlib import Path
import json
from imgtok.model import CosmoGridImageAutoEncoder, CosmoGridImageAutoEncoderAdv
import matplotlib.pyplot as plt
from ssim_psnr_iou import *
# 加载模型
from plot_plot import *


path_checkpoint = "/mnt/si0009256k6u/default/Tomasz-FHNW/CosmoGrid_Working/vqvae_update_paper/cosmogrid_adv/mxzjxs7u/checkpoints/"
ckpt1 = "epoch=0618-step=118800-val_mse=0.130104.ckpt"
ckpt_path = path_checkpoint + ckpt1
try:
    model = CosmoGridImageAutoEncoderAdv.load_from_checkpoint(ckpt_path, map_location="cpu")
except Exception:
    model = CosmoGridImageAutoEncoder.load_from_checkpoint(ckpt_path, map_location="cpu")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 读取验证集索引
with open("/mnt/si0009256k6u/default/Tomasz-FHNW/CosmoGrid_Working/vqvae_update_paper/cosmogrid_data/valid-index.json", 'r') as f:
    valid_index = json.load(f)

# 验证参数
log_mean = -4.9179
log_stddev = 0.4947

# 创建结果目录
output_dir = Path("results/")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(valid_index)} validation segments in index")

data_path = Path("/mnt/si0009256k6u/default/Tomasz-FHNW/CosmoGrid_Working/vqvae_update_paper/cosmogrid_data/content/baryonified512_036.npy")
data = np.load(data_path, mmap_mode='r')
print(data.shape)
origs = []; preds = []
for local_i in range(192):
            sample = data[local_i]
            # 保存原始数据
            original_sample = sample.copy()
            
            # 确保形状正确 (H, W) -> (1, 1, H, W)
            if sample.ndim == 2:
                sample = sample[np.newaxis, np.newaxis, :, :]
            else:
                sample = sample[np.newaxis, :, :, :]
            
            # 预处理
            x = np.clip(sample, 1e-10, None)
            x = np.log(x)
            x = (x - log_mean) / log_stddev
            x = torch.from_numpy(x).float()
            
            # 推理
            with torch.no_grad():
                x = x.cuda() if torch.cuda.is_available() else x.cpu()
                reconstructed = model.model(x)  # 使用内部codec进行重构
                
                # 反向预处理重构结果
                reconstructed_np = reconstructed.cpu().numpy()
                reconstructed_np = reconstructed_np * log_stddev + log_mean
                reconstructed_np = np.exp(reconstructed_np)
                
                orig = original_sample
                pred = reconstructed_np.squeeze()

                print(orig.shape)
                print(pred.shape)
                
                origs.append(orig)
                preds.append(pred)

origs = np.array(origs)
preds = np.array(preds)
np.save("origs.npy", origs)
np.save("preds.npy", preds)

