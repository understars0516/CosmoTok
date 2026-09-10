import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_prediction_comparison(orig, pred1, pred2, save_path=None, dpi=300):
    """
    绘制6幅对比图: orig, pred1, pred2, |pred1/orig-1|, |pred2/orig-1|, pred1/pred2
    
    Parameters:
        orig: np.ndarray, 原始图像/数据
        pred1: np.ndarray, 预测结果1
        pred2: np.ndarray, 预测结果2
        save_path: str, 保存路径(None则不保存)
        dpi: int, 输出分辨率
    """
    # ========== 1. 数据预处理 & 安全计算 ==========
    eps = 1e-8  # 防止除零
    
    # 相对误差
    err1 = (pred1 / (orig + eps) - 1.0)/10
    err2 = (pred2 / (orig + eps) - 1.0)/10
    
    # pred1/pred2 比值
    ratio = pred1 / (pred2 + eps) - 1
    
    # 统一原图与预测图的显示范围
    vmin_data = min(orig.min(), pred1.min(), pred2.min())
    vmax_data = max(orig.max(), pred1.max(), pred2.max())
    
    # 误差图上限(clip掉极端异常值，让颜色更有区分度)
    err_vmax = max(np.percentile(err1, 99), np.percentile(err2, 99))
    
    # 比值图对称范围(以1为中心)
    ratio_center = 1.0
    ratio_dev = max(abs(ratio.min() - 1.0), abs(ratio.max() - 1.0))
    ratio_dev = min(ratio_dev, 2.0)  # clip极端比值
    ratio_vmin = ratio_center - ratio_dev
    ratio_vmax = ratio_center + ratio_dev
    
    # ========== 2. 画图配置 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    #fig.suptitle('Prediction Comparison Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # 子图配置: (title, data, cmap, vmin, vmax)
    panel_configs = [
        (r'$Orig$',       orig,   'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{base}$',   pred1,  'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{adv}$',   pred2,  'viridis',  vmin_data,   vmax_data),
        (r'$Pred_{base}/Orig − 1$', err1, 'YlOrRd',  0,           err_vmax),
        (r'$Pred_{adv}/Orig − 1$', err2, 'YlOrRd',  0,           err_vmax),
        (r'$Pred_{base} / Pred_{adv}$',    ratio,'RdBu_r',  ratio_vmin,  ratio_vmax),
    ]
    
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.flat, panel_configs):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=22, pad=12)
        ax.axis('off')
        
        # 添加独立的 colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)
        
        # 比值图在center处加一条参考线标记
        #if title == r'Pred1 / Pred2':
        #    cb.set_label('Ratio (=1 means identical)', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ 图片已保存至: {save_path}")
    
    plt.show()


# ========== 3. 使用示例(替换为你的真实数据) ==========
if __name__ == '__main__':
    # 生成模拟数据用于演示
    #np.random.seed(42)
    #orig = np.random.rand(256, 256).astype(np.float32) * 100
    #pred1 = orig * (1 + np.random.randn(256, 256).astype(np.float32) * 0.1)
    #pred2 = orig * (1 + np.random.randn(256, 256).astype(np.float32) * 0.15 + 0.05)
    orig = np.load("orig.npy")[100]
    pred1 = np.load("pred_base.npy")[100]
    pred2 = np.load("pred_adv.npy")[100]
    
    plot_prediction_comparison(
        orig, pred1, pred2,
        save_path='comparison_dashboard.png'
    )
