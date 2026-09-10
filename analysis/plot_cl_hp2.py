import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 数据加载 (若无文件则生成模拟数据)
# ==========================================

def load_and_prepare_data():
    try:
        rearr = np.load("rearr_nside512.npy").astype("int")
        preds_raw = np.load("preds_base.npy")
        origs_raw = np.load("origs_base.npy")
        
        # 确保是一维数组
        if preds_raw.ndim > 1:
            preds_raw = preds_raw.reshape(-1)
        if origs_raw.ndim > 1:
            origs_raw = origs_raw.reshape(-1)
            
        preds = preds_raw[rearr]
        origs = origs_raw[rearr]
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Files not found. Generating simulated data for demonstration...")
        nside = 512
        npix = hp.nside2npix(nside)
        rearr = np.arange(npix)
        # 生成模拟谱
        al = np.arange(2, 3*nside)
        cl_sim = 1e-4 * (al/100)**(-2.5)
        origs = hp.synfast(cl_sim, nside, new=True)
        # 预测值：添加少量噪声和系统偏差
        noise = np.random.normal(0, 1e-5, npix)
        preds = origs * 1.02 + noise 
        
    return preds, origs

preds, origs = load_and_prepare_data()

# ==========================================
# 2. 计算功率谱与差值
# ==========================================

lmax = 1000
pred_cl = hp.anafast(preds, lmax=lmax)
orig_cl = hp.anafast(origs, lmax=lmax)

# 确定绘图范围
max_l_avail = len(pred_cl) - 1
l_plot = np.arange(2, min(lmax, max_l_avail) + 1)

# 计算 D_l 因子
factor = l_plot * (l_plot + 1) / (2 * np.pi)

# 计算 D_l (功率谱)
Dl_pred = factor * pred_cl[l_plot]
Dl_orig = factor * orig_cl[l_plot]

# 【关键修改】计算绝对差值，并同样乘以因子，保持量纲一致 (单位: mu K^2)
# Diff = D_l(pred) - D_l(orig)
Dl_diff = Dl_pred - Dl_orig

# ==========================================
# 3. 绘图
# ==========================================

plt.style.use('seaborn-v0_8-whitegrid')
# 设置画布，上下两图共享 X 轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'hspace': 0.05, 'height_ratios': [2, 1]})

# --- 上图：功率谱对比 (Log-Log) ---
ax1.loglog(l_plot, Dl_orig, label=r"$D_\ell^{\mathrm{orig}}$", color='black', linewidth=1.5, linestyle='-')
ax1.loglog(l_plot, Dl_pred, label=r"$D_\ell^{\mathrm{pred}}$", color='#d62728', linewidth=1.2, linestyle='--')

ax1.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
ax1.set_xlim(2, l_plot[-1])
# 自动设置 Y 轴范围
y_min = min(Dl_orig.min(), Dl_pred.min())
y_max = max(Dl_orig.max(), Dl_pred.max())
ax1.set_ylim(y_min * 0.8, y_max * 1.2)

ax1.legend(loc='best', frameon=True, framealpha=0.9)
ax1.grid(True, which="both", ls="-", alpha=0.3)

# --- 顶部角度坐标轴 ---
ax_top = ax1.twiny()
ax_top.set_xscale("log")
ax_top.set_xlim(ax1.get_xlim())
ticks_l = [2, 10, 100, 500, l_plot[-1]]
# 过滤掉超出范围的刻度
ticks_l = [t for t in ticks_l if t <= l_plot[-1]]
ticks_theta = [180.0/t for t in ticks_l]
# 格式化标签：大于1度显示度，小于1度显示角分
labels_theta = []
for t in ticks_theta:
    if t >= 1.0:
        labels_theta.append(f"{t:.1f}$^\circ$")
    else:
        labels_theta.append(f"{t*60:.0f}'")

ax_top.set_xticks(ticks_l)
ax_top.set_xticklabels(labels_theta)
ax_top.set_xlabel("Angular Scale")
ax_top.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
ax_top.spines['top'].set_visible(True)
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.spines['right'].set_visible(False)

# --- 下图：绝对差值 (Linear Y, Log X) ---
# 使用线性 Y 轴展示差值，因为差值有正负，且我们关心绝对偏差大小
ax2.semilogx(l_plot, Dl_diff, color='#d62728', linewidth=1.2, label=r"$D_\ell^{\mathrm{pred}} - D_\ell^{\mathrm{orig}}$")
ax2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.6) # 零参考线

# 设置差值图的 Y 轴标签 (单位与上图一致)
ax2.set_ylabel(r"$\Delta$")
ax2.set_xlabel(r"Multipole Moment $\ell$")

# 自动调整差值图的 Y 轴范围，留出一点边距
diff_min, diff_max = Dl_diff.min(), Dl_diff.max()
diff_range = diff_max - diff_min
# 如果差值非常小，至少给一个对称的范围或者基于最大绝对值的范围
if diff_range == 0:
    ax2.set_ylim(-1e-10, 1e-10)
else:
    margin = diff_range * 0.1
    ax2.set_ylim(diff_min - margin, diff_max + margin)

ax2.grid(True, which="both", ls="-", alpha=0.3)
# 添加图例
ax2.legend(loc='best', frameon=True, framealpha=0.9)

# 格式化 Y 轴为科学计数法，方便读取微小差值
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 隐藏上图的 X 标签
plt.setp(ax1.get_xticklabels(), visible=False)

plt.tight_layout()
save_name = "TT_cl_comparison_diff.png"
plt.savefig(save_name, bbox_inches='tight', dpi=300)
print(f"Plot saved to {save_name}")
plt.show()
