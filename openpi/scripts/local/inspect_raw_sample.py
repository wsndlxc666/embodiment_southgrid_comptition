#!/usr/bin/env python
"""看一眼 LeRobotDataset 原始样本的键名/类型/形状 —— 反事实增强要按这些键改图。

为什么需要它：`CycleCounterfactualDataset` 包在 `create_torch_dataset()` 的外面、
`transform_dataset()`（含 Repack / Normalize）的里面，所以它看到的是**原始 LeRobot 键**
（`observation.state` / `observation.images.cam_head` ...），不是 repack 之后的
`state` / `images.cam_head`。改图前必须把真实键名坐实，别照抄 repack 的键。

用法：HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data \
      /home/dgl/lxc/openpi/.venv/bin/python scripts/local/inspect_raw_sample.py
"""
from __future__ import annotations

import numpy as np

import lerobot.common.datasets.lerobot_dataset as _ld

REPO = "g1_omnipicker_clear5_d22_100ep"


def main() -> None:
    ds = _ld.LeRobotDataset(REPO, delta_timestamps={"action": [t / 20 for t in range(20)]})
    print(f"len={len(ds)}  fps={ds.fps}")
    item = ds[0]
    print(f"\n{'键':<40} {'类型':<16} {'形状':<22} dtype")
    print("-" * 90)
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            print(f"{k:<40} {type(v).__name__:<16} {str(v.shape):<22} {v.dtype}")
        elif hasattr(v, "shape"):
            print(f"{k:<40} {type(v).__name__:<16} {str(tuple(v.shape)):<22} {getattr(v, 'dtype', '?')}")
        else:
            s = repr(v)
            print(f"{k:<40} {type(v).__name__:<16} {s[:60]:<22}")

    print("\n--- 图像键的像素范围（确认是不是已归一化到 [0,1] 的 float）---")
    for k, v in item.items():
        if "image" in k:
            a = np.asarray(v)
            print(f"  {k}: shape={a.shape} dtype={a.dtype} min={a.min():.4f} "
                  f"max={a.max():.4f} mean={a.mean():.4f}")


if __name__ == "__main__":
    main()
