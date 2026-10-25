#!/usr/bin/env python
"""「目标位置」这一维到底有没有信息量？—— 直接量它，别靠感觉。

背景：用户要重启设计，Stage 2 的核心假设是「把目标位置喂给模型能提升成功率」。
但在仿真里，5 把工具的位置是**一张固定常量表**（_TOOL_REFERENCE_POS_B），
采集时只对槽位 Y 做小幅抖动。如果每集每个轮次的抓取点几乎不动，
那「目标位置」就是个近乎常数 —— 模型背下来即可，不构成新信息。

怎么量：用已有的 100 集，在每次**闭爪帧**（=该轮真正抓到工具的那一刻）记录
右臂末端位置。闭爪帧的 EE 位置就是「这把工具实际在哪」的无偏代理
（memory: 训练 EE 会走到 goals[k] 的 4~19mm 内）。

然后看两件事：
  1. **跨集**同一轮次的抓取点散布  -> 目标位置的"信息量"
  2. **集内**相邻轮次的抓取点间距  -> 区分相邻工具的"分辨率下限"

判据（预先写死，不许事后改口）：
  - 跨集 std < 5mm  -> 目标位置≈常数，喂给模型没有意义（模型记 5 个数即可）
  - 跨集 std > 20mm -> 目标位置是真信息，值得作为 Stage 2 的唯一变量
  - 中间地带      -> 信息量微弱，单靠它大概率测不出效果

只读，不写任何数据。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

REPO = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
EE = slice(7, 10)      # r_pos_x/y/z
GRIP = 16              # r_grip_inner_norm
THRESH = 0.5           # memory: norm 1.0=闭合, 0.048=张开
N_CYCLE = 5


def grasp_frames(g: np.ndarray) -> list[int]:
    """每次「张->闭」上升沿之后的第一个闭爪帧 = 该轮抓住工具的时刻。"""
    closed = g > THRESH
    rising = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
    return rising.tolist()


def main() -> None:
    files = sorted(glob.glob(os.path.join(REPO, "data/chunk-000/episode_*.parquet")))
    print(f"扫描 {len(files)} 集\n")

    # pts[cycle] = [(x,y,z), ...] 跨集收集
    pts: list[list[np.ndarray]] = [[] for _ in range(N_CYCLE)]
    n_bad = 0
    ep_intra: list[np.ndarray] = []

    for f in files:
        t = pq.read_table(f, columns=["observation.state"])
        st = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
        g = st[:, GRIP]
        ee = st[:, EE].astype(np.float64)
        gf = grasp_frames(g)
        if len(gf) != N_CYCLE:
            n_bad += 1
            continue
        picks = [ee[i] for i in gf]
        for k in range(N_CYCLE):
            pts[k].append(picks[k])
        # 集内相邻轮次间距
        ep_intra.append(np.linalg.norm(np.diff(np.stack(picks), axis=0), axis=1))

    print(f"闭爪沿==5 的集: {len(files) - n_bad} / {len(files)}"
          f"（{n_bad} 集轮次切分异常，已跳过）\n")

    print("=== 1. 跨集抓取点散布（同一轮次，100 集之间）===")
    print(f"{'轮次':<6}{'工具':<12}{'中心 xyz (base系)':<34}{'跨集 std(mm)':<16}{'跨度(mm)'}")
    print("-" * 88)
    tools = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
    all_std = []
    for k in range(N_CYCLE):
        a = np.stack(pts[k])                      # (N,3)
        mu = a.mean(axis=0)
        sd = a.std(axis=0) * 1000
        rng = (a.max(axis=0) - a.min(axis=0)) * 1000
        all_std.append(sd.max())
        print(f"{k+1:<6}{tools[k]:<12}"
              f"[{mu[0]:+.4f},{mu[1]:+.4f},{mu[2]:+.4f}]   "
              f"x{sd[0]:5.1f} y{sd[1]:5.1f} z{sd[2]:5.1f}   "
              f"x{rng[0]:5.1f} y{rng[1]:5.1f} z{rng[2]:5.1f}")

    print(f"\n  最大跨集 std = {max(all_std):.1f} mm")

    print("\n=== 2. 集内相邻轮次间距（区分相邻工具的分辨率下限）===")
    intra = np.stack(ep_intra)                    # (N, 4)
    for j in range(N_CYCLE - 1):
        col = intra[:, j]
        print(f"  第{j+1}->{j+2} 把: 中位 {np.median(col)*1000:6.1f} mm   "
              f"p10 {np.percentile(col,10)*1000:6.1f}  p90 {np.percentile(col,90)*1000:6.1f}")

    print("\n=== 判据 ===")
    m = max(all_std)
    if m < 5:
        print(f"  跨集 std {m:.1f}mm < 5mm  -> **目标位置≈常数**，喂给模型没有意义")
    elif m > 20:
        print(f"  跨集 std {m:.1f}mm > 20mm -> 目标位置是真信息，值得做 Stage 2")
    else:
        print(f"  跨集 std {m:.1f}mm 落在 5~20mm  -> 信息量微弱，单靠它大概率测不出效果")

    # 附带：模型背一张常量表就够了吗？比较"常量表残差" vs "相邻工具间距"
    print("\n=== 3. 对照：用固定常量表预测抓取点，残差有多大 ===")
    for k in range(N_CYCLE):
        a = np.stack(pts[k])
        mu = a.mean(axis=0)
        res = np.linalg.norm(a - mu, axis=1) * 1000
        print(f"  轮次{k+1} {tools[k]:<12} 残差 中位 {np.median(res):5.1f}mm  "
              f"p90 {np.percentile(res,90):5.1f}mm")


if __name__ == "__main__":
    main()
