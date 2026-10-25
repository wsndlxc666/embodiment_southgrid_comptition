#!/usr/bin/env python
"""边界之后，手臂位姿要过多久才能"看出这是第几个相位"？

用途：给加权采样器选窗口长度 K。
每个相位起点（0/366/766/1166/1566/1966）的手臂位姿在 k=0 时几乎相同（都刚放完上一件、
停在箱子上方），所以那几帧**只能靠 idx 决定去哪**。随着手臂朝各自的目标移动，位姿逐渐
分开 —— 一旦分开，模型就能靠本体感受认出相位，不再需要 idx。

所以「需要 idx 的样本」= 从边界起、到六条轨迹两两可分辨为止的那一段。
判据取位姿部分（dims 7:10）的最小两两距离 > 阈值（默认 20mm）。

只读 parquet，不碰仿真。
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

PARQUET = ("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep/"
           "data/chunk-000/episode_000000.parquet")
PHASE_START = [0, 366, 766, 1166, 1566, 1966]      # 6 个相位首帧
POS = slice(7, 10)                                  # 右末端 xyz (base 系)
K_MAX = 100


def main() -> None:
    t = pq.read_table(PARQUET, columns=["observation.state"])
    S = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
    print(f"ep0 state {S.shape}，相位首帧 {PHASE_START}")

    print(f"\n{'k':>4} {'最小两两距离mm':>14} {'中位mm':>9} {'最大mm':>9}")
    k_star = None
    for k in range(K_MAX):
        P = np.stack([S[c + k, POS] for c in PHASE_START])      # (6,3)
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        mn, md, mx = d.min() * 1000, np.median(d[d < np.inf]) * 1000, d.max() * 1000
        if k_star is None and mn > 20.0:
            k_star = k
        flag = "  ← 首次全部可分辨(>20mm)" if k == k_star else ""
        if k < 12 or k % 10 == 0 or flag:
            print(f"{k:>4} {mn:>14.1f} {md:>9.1f} {mx:>9.1f}{flag}")

    print(f"\n判据 20mm：k* = {k_star}  ⇒ 窗口长度取 K = {k_star}")
    print(f"  6 个相位 × {k_star} 帧 = {6 * k_star} 帧 / 2101 = "
          f"{100.0 * 6 * k_star / 2101:.1f}% 的样本是「必须靠 idx」的")


if __name__ == "__main__":
    main()
