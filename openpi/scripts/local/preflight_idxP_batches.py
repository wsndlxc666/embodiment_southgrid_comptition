#!/usr/bin/env python
"""端到端预检：用**真实的** create_data_loader 跑 idxP 这档，数一数每个 batch 里
各相位的样本占比，确认跟 verify_phase_weight.py 的设计值对得上。

只读数据、不建模型、不碰 GPU —— 目的是在起 14 小时训练**之前**发现"采样器接错了"
或"权重没生效"，而不是训完才发现。

必须写成 .py 文件跑（stdin 会让 DataLoader 的 spawn worker 找不到 `<stdin>`）。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/preflight_idxP_batches.py
"""
from __future__ import annotations

import collections

import numpy as np

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.training import config as _config
from openpi.training import cycle_index as CI
from openpi.training import data_loader as _data_loader

NAME = "pi05_g1_omnipicker_clear5_d22idxP_lora_20k"
NBATCH = 12
TOOL_ORDER = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
OK = True


def check(cond: bool, msg: str) -> None:
    global OK
    OK = OK and bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")


def main() -> None:
    cfg = _config.get_config(NAME)
    print(f"档 {NAME}  cycle_phase_weights={cfg.data.base_config.cycle_phase_weights}")

    loader = _data_loader.create_data_loader(
        cfg, framework="pytorch", shuffle=True, num_batches=NBATCH
    )
    # 两层包装：DataLoaderImpl -> TorchDataLoader -> torch DataLoader。
    # __iter__ 产出 (Observation, actions)；真正的 torch loader 用 .torch_loader 取。
    tl = loader._data_loader.torch_loader
    bs = tl.batch_size
    print(f"DataLoader: batch_size={bs}  sampler={type(tl.sampler).__name__}"
          f"  每轮样本数={len(tl.sampler)}")

    # 采样器的 base 数组 → 全局帧索引；每个索引对应数据集里的哪一帧，需要 layout 才能解。
    layout = CI.build_cycle_layout(
        lerobot_dataset.LeRobotDataset(cfg.data.repo_id),
        window=cfg.data.base_config.cycle_boundary_window,
        seed=0,
    )
    n_ph = max(layout.phase_sizes()) + 1
    sampler_idx = np.array(list(iter(tl.sampler)), dtype=np.int64)

    # DataLoader 是按 sampler 的序列切 batch 的（无额外打乱），所以前 NBATCH 个 batch
    # 对应的就是 sampler_idx 的前 NBATCH*batch_size 个。
    take = sampler_idx[: NBATCH * bs]
    cnt = np.bincount(layout.phase[take], minlength=n_ph)
    print(f"\n前 {NBATCH} 个 batch（{len(take)} 个样本）的相位占比:")
    for p in range(n_ph):
        who = TOOL_ORDER[p] if p < len(TOOL_ORDER) else "收尾段"
        print(f"  相位 {p} {who:<10} {cnt[p]:>5}  {100 * cnt[p] / len(take):5.1f}%")

    design = cnt / len(take)
    check(bool((design[0] > 0.10)), f"相位 0 占比 {100 * design[0]:.1f}% > 10%（加权前是 7.6%）")
    check(bool((design[2] > 0.08) and (design[3] > 0.08)),
          f"相位 2/3（当前唯一做对的两件）仍 >8%: {100 * design[2]:.1f}% / {100 * design[3]:.1f}%")
    check(bool(np.all(cnt > 0)), "六个相位都出现在 batch 里（没有哪个被饿死）")

    # 真跑一个 batch，确认 forward 需要的键都在、shape 对
    print("\n试取 1 个 batch:")
    obs, acts = next(iter(loader))
    check(tuple(acts.shape[1:]) == (20, 32),
          f"actions shape {tuple(acts.shape)} == (B, 20, 32)（20 步 × 32 维补零）")
    for k, v in obs.to_dict().items():
        shp = getattr(v, "shape", None)
        print(f"  {k:<14} {type(v).__name__:<12} {shp}")

    print("\n" + ("预检通过" if OK else "**有 FAIL，别起训**"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
