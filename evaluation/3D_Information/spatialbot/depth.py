import torch 
import numpy as np
from PIL import Image
import os


#repo name
repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_zoe_nk.to(device)

import numpy as np
from PIL import Image
import matplotlib as mpl

def save_depth_vis_minmax(
    depth, out_png,
    use_percentile=True, prct=(2, 98),
    invert=True, cmap="magma",
    save_gray16=False
):
    """
    depth: HxW 浮点深度(米或相对深度均可)
    use_percentile: True 时用分位数确定范围(更稳)，False 则用绝对 min/max
    prct: 用于分位数的 (low, high)
    invert: 是否让“近处更亮，远处更暗”
    cmap:   Matplotlib 的 colormap 名，如 'magma'/'viridis'/'turbo'
    save_gray16: 另存 16-bit 灰度(0..65535)到 out_png.replace('.png', '_gray16.png')
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if not valid.any():
        raise ValueError("depth 全是 NaN/Inf")

    if use_percentile:
        vmin = np.percentile(d[valid], prct[0])
        vmax = np.percentile(d[valid], prct[1])
    else:
        vmin = float(d[valid].min())
        vmax = float(d[valid].max())

    # 防止除零
    if not (vmax > vmin):
        vmax = vmin + 1e-6

    # 归一化到 [0,1]
    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm

    # 1) 彩色伪彩
    cmap_fn = mpl.cm.get_cmap(cmap)
    color = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)  # HxWx3
    Image.fromarray(color).save(out_png)

    # 2) 可选：16-bit 灰度(方便后处理)
    if save_gray16:
        gray16 = (norm * 65535.0).astype(np.uint16)
        Image.fromarray(gray16).save(out_png.replace(".png", "_gray16.png"))

def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)


if __name__ == "__main__":
    # Load the image
    image_path = "rgb.jpg"
    image= Image.open(image_path).convert("RGB")
    depth_numpy = model_zoe_nk.infer_pil(image)
    print(depth_numpy)
    save_raw_16bit(depth_numpy, "depth.png")

    