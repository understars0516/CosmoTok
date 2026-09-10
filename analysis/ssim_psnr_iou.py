import numpy as np

def evaluate_metrics(pred, orig, threshold=None, data_range=None, eps=1e-12):
    """
    Evaluate PSNR, SSIM (global), and IoU using only numpy.

    Parameters
    ----------
    pred : ndarray
        Reconstructed image (H, W)
    orig : ndarray
        Ground truth image (H, W)
    threshold : float or None
        Threshold for IoU binarization.
        If None, use mean of orig.
    data_range : float or None
        Dynamic range. If None, use orig.max() - orig.min().
    eps : float
        Numerical stability term.

    Returns
    -------
    psnr : float
    ssim : float
    iou  : float
    """

    pred = np.asarray(pred, dtype=np.float64)
    orig = np.asarray(orig, dtype=np.float64)

    # ------------------
    # Dynamic range
    # ------------------
    if data_range is None:
        data_range = orig.max() - orig.min()

    # ==================
    # PSNR
    # ==================
    mse = np.mean((pred - orig) ** 2)

    if mse < eps:
        psnr = np.inf
    else:
        psnr = 20 * np.log10(data_range + eps) - 10 * np.log10(mse + eps)

    # ==================
    # SSIM (Global)
    # ==================
    k1 = 0.01
    k2 = 0.03

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    mu_x = np.mean(pred)
    mu_y = np.mean(orig)

    sigma_x = np.var(pred)
    sigma_y = np.var(orig)
    sigma_xy = np.mean((pred - mu_x) * (orig - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    ssim = numerator / (denominator + eps)

    # ==================
    # IoU
    # ==================
    if threshold is None:
        threshold = np.mean(orig)

    pred_bin = pred > threshold
    orig_bin = orig > threshold

    intersection = np.logical_and(pred_bin, orig_bin).sum()
    union = np.logical_or(pred_bin, orig_bin).sum()

    iou = intersection / (union + eps)

    return psnr, ssim, iou

