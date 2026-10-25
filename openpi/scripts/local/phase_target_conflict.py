#!/usr/bin/env python
"""判决实验（不需要模型）：输入同样接近的样本，真值轨迹差多少？

这是"索引通道值多少梯度"的最直接度量。对每个 (锚点, 帧偏移 k) 的点：
  · 找**同锚点、别的集**的最近邻 → 输入差 d_same，真值轨迹差 Δ_same
    （Δ_same 就是抖动造成的不可约误差，是"位姿已定相位"时的地板）
  · 找**别的锚点、别的集**的最近邻 → 输入差 d_cross，真值轨迹差 Δ_cross
    （Δ_cross 是"位姿分不出相位"时，靠位姿再也消不掉的误差 —— 只能靠 goal_y/idx）

判据：在 d 同样小（<10mm）的前提下，若 Δ_cross ≫ Δ_same，说明
      **存在大量"输入几乎相同、输出差很远"的样本** ⇒ 索引通道有真实且大的梯度。

Δ 取 20 帧 chunk 内逐步右末端位置的**最大**距离（mm）—— 即"未来 1 秒里两条轨迹分得最开多少"。

只读 parquet。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
ANCHOR = [366, 766, 1166, 1566]     # 只取真正的变化点（不含 f0 的 home 位姿）
POS = slice(7, 10)
CH = 20                             # action_horizon
K_LIST = [0, 5, 10, 20, 30, 40, 60]
NEAR_MM = 10.0


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-*", "episode_*.parquet")))
    P, A = [], []
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "action"])
        S = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
        Ac = np.stack([np.asarray(v, dtype=np.float64) for v in t["action"].to_pylist()])
        P.append(S[:, POS])
        A.append(Ac[:, POS])
    P, A = np.stack(P), np.stack(A)          # (E,T,3) 位置 / (E,T,3) 动作里的位置
    E, NPH = P.shape[0], len(ANCHOR)
    print(f"{E} 集，变化点 {ANCHOR}，chunk {CH} 帧，位置单位 mm")

    print(f"\n{'k':>4} {'同相位: d中位':>12} {'Δ中位':>9} "
          f"{'跨相位: d<10mm占比':>16} {'Δ中位':>9} {'Δ p90':>9} {'Δ/Δ同':>7}")
    print("-" * 78)
    for k in K_LIST:
        X = np.stack([P[:, c + k, :] for c in ANCHOR], axis=1)                # (E,NPH,3)
        # 真值轨迹：从该点起 CH 帧的右末端位置 (E,NPH,CH,3)
        T = np.stack([A[:, c + k:c + k + CH, :] for c in ANCHOR], axis=1)

        # 全对全距离 (E,NPH,E,NPH)
        D = np.linalg.norm(X[:, :, None, None, :] - X[None, None, :, :, :], axis=-1)  # 米
        DT = np.linalg.norm(T[:, :, None, None] - T[None, None, :, :], axis=-1).max(-1)  # 米
        same_ep = np.eye(E, dtype=bool)[:, None, :, None]        # 同集
        same_ph = np.eye(NPH, dtype=bool)[None, :, None, :]      # 同锚点
        ee = np.arange(E)[:, None] * np.ones((1, NPH), dtype=int)
        ii = np.ones((E, 1), dtype=int) * np.arange(NPH)[None, :]

        def nearest(mask):
            flat = np.where(mask, D, np.inf).reshape(E, NPH, E * NPH)
            j = flat.argmin(-1)
            d = flat.min(-1).ravel() * 1000
            e2, i2 = j // NPH, j % NPH
            dt = DT[ee, ii, e2, i2].ravel() * 1000
            return d, dt

        ds_same, dt_same = nearest(same_ph & ~same_ep)      # 同锚点、别的集
        ds_cross, dt_cross = nearest(~same_ph & ~same_ep)   # 别的锚点、别的集

        ds_same = np.array(ds_same); dt_same = np.array(dt_same)
        ds_cross = np.array(ds_cross); dt_cross = np.array(dt_cross)
        m = ds_cross < NEAR_MM
        ms = ds_same < NEAR_MM
        print(f"{k:>4} {np.median(ds_same):>11.2f}mm {np.median(dt_same[ms]):>8.1f} "
              f"{100.0*m.mean():>15.1f}% {np.median(dt_cross[m]):>8.1f} "
              f"{np.percentile(dt_cross[m], 90):>8.1f} "
              f"{np.median(dt_cross[m])/max(np.median(dt_same[ms]), 1e-9):>6.2f}x")

    print("\n读法：Δ同 = 抖动地板（位姿已定相位时躲不掉的误差）")
    print("      Δ跨 = 位姿分不出相位时的误差，只能靠 goal_y/idx 消掉")
    print("      Δ跨/Δ同 越大、且 d<10mm 的样本越多 ⇒ 索引通道的梯度越大")


if __name__ == "__main__":
    main()
