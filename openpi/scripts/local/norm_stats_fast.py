#!/usr/bin/env python
"""官方 compute_norm_stats 的等价加速版。

走**完全相同**的代码路径（同一个 dataset / transform / RunningStats / 批顺序），
只把视频解码整个跳过 —— norm 统计只累计 state/actions，图片对结果没有任何影响，
但官方脚本要为每个样本解 20 步 × 2 路 HEVC 视频。

用法:
    norm_stats_fast.py <config-name>                 # 正常算
    norm_stats_fast.py <config-name> --verify <json> # 算完与已有 json 逐维对比

⚠ 曾踩的坑（2026-09-22）
------------------------
第一版 stub 返回 `torch.zeros((len(query_ts), 3, 480, 640), uint8)` —— 形状对了，
但**每个样本 37MB**（20×3×480×640×2 相机）。380281 个样本 = 14 TB 内存写入，
实测烧了 11 分钟只推进一小截，看着像"卡住"其实是内存带宽打满。

下游 `_parse_image` **不做 resize**，图片形状只是原样塞进 batch dict，
而 norm 统计根本不读图片。所以形状可以随便缩 —— 这里用 8×8，
同样的正确性，每样本 7.7KB，快约 4800 倍。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# 只为让 batch 里有个形状一致的小张量；norm 统计不读它，无需真实分辨率。
_VIDEO_SHAPE = (3, 8, 8)


def _query_videos_stub(self, query_timestamps, ep_idx):
    """返回小零张量，不做任何解码。形状与原实现 .squeeze(0) 之后一致。"""
    return {
        vid_key: torch.zeros((len(query_ts), *_VIDEO_SHAPE), dtype=torch.uint8)
        for vid_key, query_ts in query_timestamps.items()
    }


lerobot_dataset.LeRobotDataset._query_videos = _query_videos_stub

# ---------------- 参数 ----------------
argv = sys.argv[1:]
config_name = argv[0]
verify_path = None
if "--verify" in argv:
    verify_path = argv[argv.index("--verify") + 1]

sys.argv = ["compute_norm_stats.py", "--config-name", config_name]

import runpy  # noqa: E402

runpy.run_path("/home/dgl/lxc/openpi/scripts/compute_norm_stats.py", run_name="__main__")

# ---------------- 校验 ----------------
if verify_path:
    import openpi.training.config as C
    from openpi.shared import normalize

    cfg = C.get_config(config_name)
    new_path = cfg.assets_dirs / cfg.data.repo_id
    new = normalize.load(new_path)
    old = normalize.load(verify_path)

    print(f"\n===== 与 {verify_path} 对比 =====")
    bad = 0
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            a = np.asarray(getattr(old[key], field))
            b = np.asarray(getattr(new[key], field))
            if a.shape != b.shape:
                print(f"✗ {key}.{field} 维度不同: {a.shape} vs {b.shape}")
                bad += 1
                continue
            d = np.abs(a - b)
            # 分位数是 5000 桶直方图出来的，桶宽 = (max-min)/5000，容差给 2 桶
            rel = float(np.max(d / np.maximum(np.abs(a), 1e-3)))
            print(f"  {key}.{field:<5} 最大绝对差 {d.max():.3e}  最大相对差 {rel:.3e}")
            if rel > 1e-3:
                i = int(np.argmax(d / np.maximum(np.abs(a), 1e-3)))
                print(f"      ↑ 最差在第 {i} 维: 旧 {a[i]:.6f} 新 {b[i]:.6f}")
                bad += 1
    print("✓ 等价" if bad == 0 else f"✗ 有 {bad} 处不一致")
    sys.exit(1 if bad else 0)
