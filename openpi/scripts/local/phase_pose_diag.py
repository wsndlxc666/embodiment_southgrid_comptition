#!/usr/bin/env python
"""诊断：相位首帧的位姿云到底分不分得开，goal_y 到底随相位变不变。"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
PS = [0, 366, 766, 1166, 1566]


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-*", "episode_*.parquet")))
    P, GY, GD, IX = [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=["observation.state"])
        S = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
        P.append(S[:, 7:14])
        GY.append(S[:, 19])
        GD.append(S[:, 18])
        IX.append(S[:, 21])
    P, GY, GD, IX = np.stack(P), np.stack(GY), np.stack(GD), np.stack(IX)
    print("state 位姿", P.shape)

    print("\n相位首帧(k=0) 各维: goal_y / goal_dist / cycle_idx")
    for i, c in enumerate(PS):
        print(f"  ph{i} f{c:<5} goal_y={GY[:, c].mean():+.4f}(std {GY[:, c].std():.4f}) "
              f"goal_dist={GD[:, c].mean():.4f} idx={IX[:, c].mean():.1f}")

    print("\n各相位首帧 goal_y 的取值:")
    for i, c in enumerate(PS):
        print(f"  ph{i}: {np.unique(np.round(GY[:, c], 4)).tolist()}")

    print("\n相位首帧位置(pos) 云中心 / 半径:")
    C = []
    for i, c in enumerate(PS):
        p = P[:, c, 0:3]
        C.append(p.mean(0))
        r = np.median(np.linalg.norm(p - p.mean(0), axis=1)) * 1000
        print(f"  ph{i} 中心 {np.round(p.mean(0), 4)}  中位半径 {r:6.1f}mm")
    C = np.stack(C)
    dd = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1) * 1000
    print("\n云中心两两距离 (mm):")
    for i in range(5):
        print("   " + " ".join(f"{dd[i, j]:7.1f}" for j in range(5)))

    print("\n7 维(含四元数) 跨集两两距离 (mm):")
    for i in range(5):
        for j in range(i + 1, 5):
            d = np.linalg.norm(P[:, PS[i], :][:, None, :] - P[:, PS[j], :][None, :, :], axis=-1) * 1000
            print(f"  ph{i}-ph{j}: 中位 {np.median(d):7.2f}  最小 {d.min():7.2f}  最大 {d.max():7.2f}")

    print("\n同相位跨集距离 (mm) —— 这就是抖动带宽度:")
    for i, c in enumerate(PS):
        d = np.linalg.norm(P[:, c, :][:, None, :] - P[:, c, :][None, :, :], axis=-1) * 1000
        d[d == 0] = np.nan
        print(f"  ph{i}: 中位 {np.nanmedian(d):6.2f}  最大 {np.nanmax(d):6.2f}")


if __name__ == "__main__":
    main()
