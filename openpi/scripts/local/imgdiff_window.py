#!/usr/bin/env python
"""闭爪前后 cam_head 的逐帧图像变化量:看模型有没有可用的视觉前兆。"""
import glob
import numpy as np
from PIL import Image

fs = sorted(glob.glob("/tmp/hf/*.png"))
ims = [np.asarray(Image.open(f).convert("L").resize((160, 90)), dtype=np.float32) for f in fs]
d = [np.abs(ims[i] - ims[i - 1]).mean() for i in range(1, len(ims))]

print("帧号  相邻帧图像差异(cam_head)          [闭爪在 113->114]")
for i, v in enumerate(d):
    fr = 70 + i + 1
    mark = "  <== 闭爪" if fr == 114 else ""
    print(f"  {fr:>4} {v:>6.2f} {'#' * int(v * 4)}{mark}")

print(f"\n全窗口均值 = {np.mean(d):.2f}")
pre = [v for i, v in enumerate(d) if 85 <= 70 + i + 1 <= 113]
print(f"闭爪前静止段(85~113)均值 = {np.mean(pre):.2f}")
print(f"闭爪那一帧(113->114)   = {d[114 - 70 - 1]:.2f}")
