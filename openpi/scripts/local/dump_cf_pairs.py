#!/usr/bin/env python
"""把改法 A 的"参照帧 vs 供体帧"拼成对照图，用眼睛确认换过去的图像是不是合理的输入。

为什么要看：头相机(cam_head)会拍到机器人**自己的手臂**。我们保留参照帧的臂位姿、
却换成供体帧的图像 —— 如果供体那帧里手臂在别的位置，图像就和本体感觉自相矛盾，
模型可能学成"看图像里手臂在哪"而不是"看还剩哪些工具"，那就白训了。
这一眼决定要不要给供体加约束（比如只在同相位阶段的帧里挑）。

用法：HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data \\
      /home/dgl/lxc/openpi/.venv/bin/python scripts/local/dump_cf_pairs.py
输出：/home/dgl/lxc/.runlogs/cf_pairs/*.png
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

import openpi.training.config as _config
import openpi.training.cycle_index as _ci
import openpi.training.data_loader as _dl

CONFIG = "pi05_g1_omnipicker_clear5_d22idxI_lora_20k"
OUT = pathlib.Path("/home/dgl/lxc/.runlogs/cf_pairs")
# (参照帧, 说明) —— 都在 ep0。366 是相位2的第一帧（offset 0），566 是相位2的中段。
# 窗口只有 60 帧（变化点之后 [s, s+60)），所以 offset 只能取 0..59，别取 100/200。
# 右臂在 d22 里是 dims 7:10 —— dims 0:6 是**左臂**（本任务全程不动，拿它算"EE 差"会得出
# 恒等于 1mm 的假结论，这里踩过一次）。
RARM = slice(7, 10)
PAIRS = [(366, "相位2 off0"), (396, "相位2 off30"), (416, "相位2 off50"),
         (766, "相位3 off0"), (1166, "相位4 off0"), (1566, "相位5 off0")]


def to_hwc(t) -> np.ndarray:
    a = np.asarray(t, dtype=np.float32)
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _config.get_config(CONFIG)
    data_cfg = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model))
    raw = _dl.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    layout = _ci.build_cycle_layout(raw, window=60, seed=0)
    states = _ci._load_states(_ci.find_lerobot_dataset(raw))

    from PIL import Image, ImageDraw

    rows = []
    for i, note in PAIRS:
        if not layout.boundary[i]:
            print(f"跳过 {i}：不是边界帧")
            continue
        j = int(layout.donor_frame[i])
        ref, don = raw[i], raw[j]
        pi, pj = int(states[i][_ci.CYCLE_IDX_DIM]), int(states[j][_ci.CYCLE_IDX_DIM])
        dy = float(states[i][_ci.GOAL_Y_DIM] - states[j][_ci.GOAL_Y_DIM]) * 1000
        d_arm = np.linalg.norm(np.asarray(states[i])[RARM] - np.asarray(states[j])[RARM]) * 1000
        ref_head, don_head = (np.asarray(ref["observation.images.cam_head"], dtype=np.float32),
                              np.asarray(don["observation.images.cam_head"], dtype=np.float32))
        d_img = float(np.abs(ref_head - don_head).mean()) * 255
        print(f"i={i:<5}{note:<12} 相位{pi} → 供体 j={j:<5} 相位{pj}   "
              f"goal_y 差 {dy:+6.0f}mm   右臂位置差 {d_arm:6.1f}mm   头图平均像素差 {d_img:5.1f}/255")
        tiles = [to_hwc(ref["observation.images.cam_head"]),
                 to_hwc(don["observation.images.cam_head"]),
                 to_hwc(ref["observation.images.cam_wrist_r"]),
                 to_hwc(don["observation.images.cam_wrist_r"])]
        labels = [f"ref head f{i} idx{pi}", f"DONOR head f{j} idx{pj}",
                  f"ref wrist f{i}", f"wrist f{j}"]
        strip = np.concatenate([np.concatenate(tiles[:2], axis=1),
                                np.concatenate(tiles[2:], axis=1)], axis=0)
        im = Image.fromarray(strip)
        d = ImageDraw.Draw(im)
        d.text((6, 6), labels[0], fill=(255, 0, 0))
        d.text((640 + 6, 6), labels[1], fill=(255, 0, 0))
        d.text((6, 480 + 6), labels[2], fill=(255, 0, 0))
        d.text((640 + 6, 480 + 6), labels[3], fill=(255, 0, 0))
        p = OUT / f"pair_{i:04d}_to_{j:04d}.png"
        im.save(p)
        rows.append(str(p))
    print("\n写出：")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
