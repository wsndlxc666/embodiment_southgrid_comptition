#!/usr/bin/env python3
"""量相位段拼接处的位姿缝隙：决定重排数据集要不要做平滑。

背景：要打破"相位=节拍器"（100/100 集 cycle_idx 跳变帧完全相同），唯一的办法是
把 5 个相位段重新排序。但重排会在拼接处产生位姿跳变，跳变多大会变成"教模型在
相位切换时抖一下"。

这个脚本不改任何东西，只回答一个问题：**把 seg_i 的尾巴接到 seg_j 的头上，末端
位置差多少毫米**，以及允许前后各裁 ±N 帧的话最优缝隙是多少。

用法: measure_splice_gap.py [episode_parquet]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

D = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
EE = slice(7, 10)

# cycle_idx 跳变帧实测：100/100 集完全相同（见 2026-09-20 的核查）
BOUND = [0, 366, 766, 1166, 1566, 1966, 2101]
NSEG = 6                     # seg0..seg4 = 五个工具；seg5 = 收尾回 home


def load(path: str) -> np.ndarray:
    df = pd.read_parquet(path)
    x = df["observation.state"].to_numpy()
    if x.dtype == object:
        x = np.stack([np.asarray(v, dtype=np.float64) for v in x])
    return np.asarray(x, dtype=np.float64)


def main() -> int:
    pat = sys.argv[1] if len(sys.argv) > 1 else f"{D}/data/chunk-000/episode_000000.parquet"
    files = sorted(glob.glob(pat)) or [pat]
    s = load(files[0])
    print(f"读 {files[0]}  state {s.shape}")

    # 每段的头尾各取若干帧候选，找缝隙最小的 (裁尾 a 帧, 裁头 b 帧)
    TRIM = 12
    print(f"\n各段（seg0..seg4=五件工具, seg5=收尾）实际长度:")
    for k in range(NSEG):
        seg = s[BOUND[k]:BOUND[k + 1]]
        print(f"  seg{k}: 帧[{BOUND[k]:5d},{BOUND[k+1]:5d})  长 {len(seg):4d}"
              f"  EE 首 {np.round(seg[0, EE], 3)}  末 {np.round(seg[-1, EE], 3)}"
              f"  爪首 {seg[0,16]:.4f} 爪末 {seg[-1,16]:.4f}")

    print(f"\n★ 拼接缝隙（末段裁尾 a ∈[0,{TRIM}]，头段裁头 b ∈[0,{TRIM}]，取最优）")
    print(f"  {'from→to':>10s} {'无裁剪缝隙mm':>13s} {'最优a':>6s} {'最优b':>6s}"
          f" {'最优缝隙mm':>11s}")
    worst = []
    for i in range(NSEG):
        for j in range(NSEG):
            if i == j:
                continue
            tail_all = s[BOUND[i]:BOUND[i + 1], EE]
            head_all = s[BOUND[j]:BOUND[j + 1], EE]
            direct = float(np.linalg.norm(tail_all[-1] - head_all[0]) * 1000)
            best, ba, bb = 1e9, 0, 0
            for a in range(TRIM + 1):
                te = tail_all[len(tail_all) - 1 - a]
                for b in range(TRIM + 1):
                    d = float(np.linalg.norm(te - head_all[b]) * 1000)
                    if d < best:
                        best, ba, bb = d, a, b
            flag = "" if best < 30 else ("  ← 大" if best < 80 else "  ← 很大")
            print(f"  seg{i}→seg{j:<4d} {direct:13.1f} {ba:6d} {bb:6d} {best:11.1f}{flag}")
            worst.append((best, i, j, ba, bb))

    worst.sort(reverse=True)
    print(f"\n最差的 5 对（最优裁剪后仍有这么大缝隙）：")
    for d, i, j, a, b in worst[:5]:
        print(f"  seg{i}→seg{j}: {d:6.1f} mm  (裁尾{a} 裁头{b})")

    # 段内 EE 速度，作为"正常速度"的标尺
    v = np.linalg.norm(np.diff(s[:, EE], axis=0), axis=1) * 1000 * 20   # mm/s
    print(f"\n正常段内末端速度：中位 {np.median(v):6.1f} mm/s   p95 {np.percentile(v,95):6.1f}"
          f"   最大 {v.max():7.1f} mm/s")
    print(f"⇒ 一帧内跳 {worst[0][0]:.0f} mm 等效瞬时速度 {worst[0][0]*20:.0f} mm/s"
          f"（是 p95 的 {worst[0][0]*20/np.percentile(v,95):.1f} 倍）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
