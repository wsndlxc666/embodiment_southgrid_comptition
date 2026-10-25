#!/usr/bin/env python
"""有多大比例的样本，光靠手臂位姿分不出相位？——留一集 1-近邻相位分类。

⚠ 前一版把 pos(米) 和 quat(无量纲) 混进同一个 norm，那些"78mm"其实是四元数分量差，
   不是长度（同相位跨集也能算出 100mm+）。这里只用**位置**，单位就是毫米。

对每个 (相位 i, 留出的集 e, 帧偏移 k)，在**其它 99 集**的全部相位里找位置最近邻；
若最近邻属于别的相位且距离 < 10mm（≈模型精度）⇒ 这个位姿**分不出相位**，
模型只能靠 goal_y / cycle_idx 消歧。

报的头两个数是决定性的：
  · 歧义比例 —— 索引通道能拿到的梯度占比（和边界的 4.76% 是同一类量）
  · 歧义样本的目标差 —— 这些样本"猜错相位"要付多大代价

只读 parquet。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
# 6 个相位首帧。0 是本集起点(在 home)，其余 5 个都是"刚放完上一件"的变化点
ANCHOR = [0, 366, 766, 1166, 1566, 1966]
POS = slice(7, 10)          # 右末端 xyz (base 系)，单位米
K_LIST = [0, 1, 2, 5, 10, 20, 30, 40, 60, 100]
NEAR_MM = 10.0
BIG_MM = 50.0               # "目标差这么大 ⇒ 猜错相位代价显著"


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-*", "episode_*.parquet")))
    P, GY = [], []
    for f in files:
        t = pq.read_table(f, columns=["observation.state"])
        S = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
        P.append(S[:, POS])
        GY.append(S[:, 19])
    P, GY = np.stack(P), np.stack(GY)
    E, NPH = P.shape[0], len(ANCHOR)
    print(f"{E} 集，锚点帧 {ANCHOR}，位姿取 dims7~9(右末端 xyz)")
    print(f"判据：位置最近邻是异相位 且 距离 < {NEAR_MM}mm ⇒ 歧义\n")

    hdr = (f"{'k':>4} {'1-NN准确率':>9} {'最近邻距离中位':>13} {'歧义比例':>9} "
           f"{'其中目标差>50mm':>14} {'歧义目标差中位':>13}")
    print(hdr)
    print("-" * len(hdr))
    for k in K_LIST:
        # (E, NPH, 3) → reshape 是 ep 主序：每集 NPH 个锚点连着
        X = np.stack([P[:, c + k, :] for c in ANCHOR], axis=1).reshape(-1, 3)
        gy = np.stack([GY[:, c + k] for c in ANCHOR], axis=1).reshape(-1)
        ep = np.repeat(np.arange(E), NPH)
        lab = np.tile(np.arange(NPH), E)

        corr = 0
        nd, nl, ng = [], [], []
        for e in range(E):
            m = ep == e
            A, Bm = X[m], ~m
            B = X[Bm]
            d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)      # (NPH, |B|)
            j = d.argmin(1)
            corr += int((lab[Bm][j] == lab[m]).sum())
            nd.append(d[np.arange(len(j)), j])
            nl.append(lab[Bm][j])
            ng.append(gy[Bm][j])
        acc = corr / (NPH * E)
        dmm = np.concatenate(nd) * 1000
        amb = (dmm < NEAR_MM) & (np.concatenate(nl) != lab)
        dg = np.abs(gy - np.concatenate(ng))[amb] * 1000
        print(f"{k:>4} {acc*100:>8.1f}% {np.median(dmm):>12.2f}mm "
              f"{100.0*amb.mean():>8.2f}% "
              f"{(100.0*np.mean(dg > BIG_MM) if amb.any() else 0.0):>13.1f}% "
              f"{(np.median(dg) if amb.any() else 0.0):>12.1f}mm")

    print("\n读法：歧义比例高 + 歧义目标差大 ⇒ 模型必须读 goal_y/cycle_idx 才能消歧，")
    print("      索引通道本来就有大梯度，加权/遮掩是在把这个梯度从噪声里抬出来；")
    print("      歧义比例低 ⇒ 位姿已是充分统计量，怎么做数据都改不了。")


if __name__ == "__main__":
    main()
