#!/usr/bin/env python
"""本体感受到底能不能唯一确定相位？——跨 100 集算「位姿云是否重叠」。

phase_divergence.py 只在 ep0 里比了六个相位起点的位姿（最小间隔 30.3mm）。但跨集有
布局抖动（x±8mm、y±18mm、yaw±6°，12% 概率一件歪到 ±40mm），所以真正的问题是：

    在**全部 100 集**里，「刚放完第 i 件」的手臂位姿和「刚放完第 j 件」的位姿，云与云之间
    的最小距离是多少？

若最小距离 ≫ 模型精度（实测 mid-cycle 7~10mm）⇒ 位姿是充分统计量，相位不需要 idx，
   模型有捷径可走，idx 拿不到梯度（这就是斜率 0.09 的根因）。
若最小距离 ≲ 精度，甚至为 0 ⇒ 存在"同一个位姿、不同相位"的样本 ⇒ 模型**必须**读
   goal_y/idx 才能消歧，索引通道有活路。

顺带报：那对最接近的样本各自的**目标**（goal_y）差多少 —— 这才是"必须读索引"的直接证据。

只读 parquet。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
PHASE_START = [0, 366, 766, 1166, 1566]        # 相位 1..5 的首帧（idx 1..5，会去抓）
POS = slice(7, 10)                              # 右末端 xyz (base 系)
K_MAX = 40


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-*", "episode_*.parquet")))
    P = []      # 只留右末端位置，省内存
    for f in files:
        t = pq.read_table(f, columns=["observation.state"])
        S = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
        P.append(S[:, POS])
    P = np.stack(P)                       # (E, T, 3)
    E = P.shape[0]
    print(f"{E} 集，state 右末端位置 {P.shape}")

    # goal_y 逐集也应一致，取来当"目标"
    t = pq.read_table(files[0], columns=["observation.state"])
    S0 = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
    goal_y = S0[:, 19]

    print(f"\n{'k':>4} {'云间最小距离mm':>15} {'该对的目标差mm':>15}   {'重叠?':>6}")
    overlaps = []
    for k in range(K_MAX):
        clouds = [P[:, c + k, :] for c in PHASE_START]        # 每个相位 (E,3)
        best = (np.inf, None, None)
        for i in range(len(clouds)):
            for j in range(i + 1, len(clouds)):
                d = np.linalg.norm(clouds[i][:, None, :] - clouds[j][None, :, :], axis=-1)
                mn = d.min() * 1000
                if mn < best[0]:
                    best = (mn, i, j)
        mn, i, j = best
        # 最接近的那对样本的目标差
        d = np.linalg.norm(clouds[i][:, None, :] - clouds[j][None, :, :], axis=-1)
        a, b = np.unravel_index(d.argmin(), d.shape)
        dg = abs(goal_y[PHASE_START[i] + k] - goal_y[PHASE_START[j] + k]) * 1000
        ov = "是" if mn <= 15.0 else ""
        if ov:
            overlaps.append((k, mn))
        if k < 12 or k % 5 == 0 or ov:
            print(f"{k:>4} {mn:>15.1f} {dg:>15.1f}   {ov:>6}")

    print(f"\n判据 15mm（≈2 倍模型精度）：重叠的 k 有 {len(overlaps)} 个"
          f"{'：' + str([o[0] for o in overlaps]) if overlaps else ''}")
    if not overlaps:
        print("  ⇒ 跨集也不重叠：**手臂位姿是相位的充分统计量**，训练里没有"
              "「同一位姿、不同目标」的样本。\n"
              "     结论：模型有捷径可走，idx 拿不到梯度是数据的结构性结果，"
              "过采样/遮掩都改不了这一点。")
    else:
        print("  ⇒ 存在「同一位姿、不同相位」的样本 ⇒ 索引通道有活路，"
              "过采样这些 k 是有意义的。")


if __name__ == "__main__":
    main()
