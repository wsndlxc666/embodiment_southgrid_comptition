#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""免解视频的 norm_stats 快速计算(口径与 openpi 一致)。

openpi 的归一化只统计 state/actions(图像不参与归一化),而 G1OmnipickerInputs
对 state/actions 是原样透传(仅改名)。因此不必 decode 全部 HEVC 帧,直接 pooled
读 train repo 所有 parquet 的 observation.state / action 列即可:

- mean/std    = E、sqrt(E[x^2]-E[x]^2)(总体,与 RunningStats 一致)
- q01/q99     = 逐维精确分位数(openpi 用 5000 bin 直方图近似,这里用精确值,更稳)

用法: python compute_pooled_norm_stats.py
写出: /home/dgl/lxc/openpi/assets/pi05_g1_omnipicker_clear5_lora/<repo_id>/norm_stats.json
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = "/home/dgl/lxc/openpi/data"
REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "g1_omnipicker_clear5_train"
ASSET_ROOT = "/home/dgl/lxc/openpi/assets/pi05_g1_omnipicker_clear5_lora"

REPO = os.path.join(DATA_ROOT, REPO_ID)
CHUNK = "chunk-000"
STATE_COL = "observation.state"
ACTION_COL = "action"


def main():
    pq_files = sorted(glob.glob(os.path.join(REPO, "data", CHUNK, "episode_*.parquet")))
    if not pq_files:
        raise SystemExit(f"no parquet under {REPO}/{CHUNK}")

    states = []
    actions = []
    frames = 0
    for f in pq_files:
        t = pq.read_table(f, columns=[STATE_COL, ACTION_COL])
        states.append(np.asarray(t.column(STATE_COL).to_pylist(), dtype=np.float32))
        actions.append(np.asarray(t.column(ACTION_COL).to_pylist(), dtype=np.float32))
        frames += t.num_rows

    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    print(f"pooled {len(pq_files)} episodes / {state.shape[0]} frames, state_dim={state.shape[1]} action_dim={action.shape[1]}")

    def stats(a):
        return {
            "mean": a.mean(axis=0),
            "std": np.sqrt(np.maximum(0.0, np.mean(a**2, axis=0) - np.mean(a, axis=0) ** 2)),
            "q01": np.quantile(a, 0.01, axis=0),
            "q99": np.quantile(a, 0.99, axis=0),
        }

    norm_stats = {"norm_stats": {"state": stats(state), "actions": stats(action)}}

    # 转成与 openpi serialize_json 相同的形状:嵌套 dict -> 数组转 list
    def to_plain(d):
        return {k: np.asarray(v).reshape(-1).tolist() for k, v in d.items()}

    payload = {"norm_stats": {k: to_plain(v) for k, v in norm_stats["norm_stats"].items()}}
    out_dir = os.path.join(ASSET_ROOT, REPO_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print("state  mean 前6:", np.round(np.asarray(payload["norm_stats"]["state"]["mean"])[:6], 4))
    print("state  q01/q99 维度数:", len(payload["norm_stats"]["state"]["q01"]), "/", len(payload["norm_stats"]["state"]["q99"]))
    print(f"-> {out_path}")

    # 交叉校验:期望恒定的维(如末端夹爪)std≈0、q01≈q99≈该常量
    s = payload["norm_stats"]["state"]
    const_dims = [(i, m, q1, q9) for i, (m, q1, q9) in
                  enumerate(zip(s["mean"], s["q01"], s["q99"])) if abs(q9 - q1) < 1e-6]
    print("恒定维(mean≈q01≈q99):", const_dims[:6], "共", len(const_dims))


if __name__ == "__main__":
    main()
