import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_astronomy_comparison(orig, pred, psnr):
    """
    天文专用绘图：仅依赖 orig, pred, psnr。
    - PSNR > 40 才画图
    - Colorbar 强制只显示 5 个刻度
    - 自动保存到 ./astro_plots
    """
    
    # 1. 阈值检查
    #:if psnr <= 20.0:
    #@    return None

    # 2. 数据准备
    orig = np.squeeze(orig)
    pred = np.squeeze(pred)
    residual = (orig - pred)#/orig
    
    # 自适应对比度 (截断 1% - 99.5%)
    v_min = np.percentile(orig, 2.0)
    v_max = np.percentile(orig, 98)
    vmin=0; vmax = 0.03
    if v_max <= v_min: v_min, v_max = orig.min(), orig.max()
    
    # 残差对称范围
    v_abs = 0.003; np.max(np.abs(residual))
    if v_abs == 0: v_abs = 1e-9
    
    # 3. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
    kwargs = {'interpolation': 'nearest', 'origin': 'lower'}
    
    # --- 子图 1: Original ---
    im1 = axes[0].imshow(orig, cmap='magma', vmin=v_min, vmax=v_max, **kwargs)
    axes[0].set_title('Original', fontsize=18, fontweight='bold')
    axes[0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, shrink=0.8)
    cbar1.locator = plt.MaxNLocator(nbins=5)  # ⭐ 强制 5 个刻度
    cbar1.update_ticks()

    # --- 子图 2: Predicted ---
    im2 = axes[1].imshow(pred, cmap='magma', vmin=v_min, vmax=v_max, **kwargs)
    axes[1].set_title(f'Predicted', fontsize=18, fontweight='bold')
    axes[1].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, shrink=0.8)
    cbar2.locator = plt.MaxNLocator(nbins=5)  # ⭐ 强制 5 个刻度
    cbar2.update_ticks()

    # --- 子图 3: Residual ---
    im3 = axes[2].imshow(residual, cmap='RdBu_r', vmin=-v_abs, vmax=v_abs, **kwargs)
    axes[2].set_title(f'Residual', fontsize=18, fontweight='bold')
    axes[2].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, shrink=0.8)
    cbar3.locator = plt.MaxNLocator(nbins=5)  # ⭐ 强制 5 个刻度
    cbar3.update_ticks()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 4. 自动保存
    save_dir = Path("./astro_plots")
    save_dir.mkdir(exist_ok=True)
    filename = f"result_psnr{psnr:.2f}.png"
    
    plt.savefig(save_dir / filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    
    print(f"✅ Saved: {save_dir / filename} (Colorbars limited to 5 ticks)")
    return True
