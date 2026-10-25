#!/usr/bin/env python
"""端到端：用真实 TrainConfig 建 loader，确认 cycle 开关接上了、批次里维度和成分都对。

对照跑两次：d22 原档（应完全不受影响）与 d22idx（应打印采样器/反事实日志）。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_cycle_loader_e2e.py
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _dl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def run(name: str, n_batches: int = 3) -> None:
    cfg = _config.get_config(name)
    cfg = dataclasses.replace(cfg, num_workers=0)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f"\n===== {name} =====")
    print(f"  cycle_boundary_window={dc.cycle_boundary_window} "
          f"weight={dc.cycle_boundary_weight} cf_prob={dc.cycle_counterfactual_prob}")
    loader = _dl.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=n_batches)
    n = 0
    n_bad = 0
    n_all = 0
    for batch in loader:
        obs, actions = batch
        st = np.asarray(obs.state)
        act = np.asarray(actions)
        if n == 0:
            print(f"  state {st.shape}  action {act.shape}")
        # PadStatesAndActions 把 state 也补到 action_dim=32（真实 22 维在前，后面 10 个 0）
        assert st.shape[1] == 32, f"state 维数 {st.shape[1]} != 32"
        assert np.allclose(st[:, 22:], 0), "第 22 维之后应该是零填充"
        assert act.shape[1:] == (20, 32), f"action 形状 {act.shape[1:]} != (20, 32)"
        # 机制断言：action[t] == state[t+1] 逐字节相等，而 goal_y 在一个相位内恒定，
        # 所以**正常**样本里 state 的 dim19 必等于 action 首步的 dim19。
        # 反事实命中时两者就断了 —— 这正是"输入说去别处、真值说去这里"。
        mismatch = np.abs(st[:, 19] - act[:, 0, 19]) > 1e-3
        n_bad += int(mismatch.sum())
        n_all += st.shape[0]
        if n == 0:
            print(f"  索引维在填充后仍是 19/20/21: "
                  f"goal_y[19]={np.round(st[:,19],3).tolist()} "
                  f"cycle_idx[21]={np.round(st[:,21],3).tolist()}")
        n += 1
    print(f"  {n} 个 batch / {n_all} 个样本正常产出；"
          f"state[19] != action[0][19] 的比例 = {100.0*n_bad/n_all:.2f}%")


if __name__ == "__main__":
    run("pi05_g1_omnipicker_clear5_d22_lora_20k", n_batches=40)
    run("pi05_g1_omnipicker_clear5_d22idx_lora_20k", n_batches=40)
    print("\nOK：两档都能建 loader，原档未受影响")
