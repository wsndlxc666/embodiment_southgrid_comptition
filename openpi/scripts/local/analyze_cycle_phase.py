#!/usr/bin/env python
"""分析 clear5 训练集的抓取周期结构，对比三种"距离维"定义的形状。

背景
----
d19 的第 19 维现在是  d(f) = min_k ||ee(f) - refs_k||，即"到最近抓取点的距离"。
问题：min 抹掉了"是哪一把"。统计整个 episode 的 argmin，工具1 当过最近 1264 帧
（帧 47~2051），根本不跟着从左到右走；而且搬运时机械臂在盒子上方，这个数只反映
"盒子碰巧离哪把工具最近"，没有意义。

本脚本先只读分析，量三件事：
  1. 181 集的闭爪周期边界是否一致（能否用统一帧号切分）
  2. 三种距离定义各自的形状、连续性、取值分布
     A) min_k      —— 现状
     B) refs[k(f)] —— 当前周期对应的那把工具（用户提的"周期版"）
     C) 目标点集合 —— 工具抓取点 ∪ 盒子落点，再取 min（天然连续）
  3. 各自的"跳变"有多大（相邻帧差值的分布）

只读 parquet，不碰视频，不碰仿真。
"""
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
EE_SLICE = slice(7, 10)      # 右臂末端 xyz (base 系)
GRIP_IDX = 16                # r_grip_inner_norm，闭爪时为 1.0，开爪 0.048
GRIP_THRESH = 0.5
N_EP = 181


def load_episode(ep: int) -> np.ndarray:
    p = os.path.join(DATA, f"data/chunk-000/episode_{ep:06d}.parquet")
    return np.array(pq.read_table(p)["observation.state"].to_pylist(), dtype=np.float32)


def find_cycles(state: np.ndarray) -> tuple[list[int], list[int]]:
    """返回 (闭爪帧, 开爪帧)。由夹爪二值方波直接求出，不做任何硬编码。"""
    closed = state[:, GRIP_IDX] > GRIP_THRESH
    edges = np.where(np.diff(closed.astype(np.int8)) != 0)[0] + 1
    grasp = [int(f) for f in edges if closed[f]]
    release = [int(f) for f in edges if not closed[f]]
    return grasp, release


def main() -> None:
    # ---------- 1. 周期边界一致性 ----------
    print("=" * 78)
    print("1. 181 集的闭爪/开爪帧号是否一致")
    print("=" * 78)

    all_grasp, all_release, lengths = [], [], []
    for ep in range(N_EP):
        s = load_episode(ep)
        g, r = find_cycles(s)
        all_grasp.append(g)
        all_release.append(r)
        lengths.append(len(s))

    shapes = {len(g) for g in all_grasp}
    print(f"  每集帧数: {min(lengths)} ~ {max(lengths)}")
    print(f"  每集闭爪次数: {sorted(shapes)}  ({'全部一致' if len(shapes) == 1 else '>>> 不一致 <<<'})")

    n_cyc = min(len(g) for g in all_grasp)
    garr = np.array([g[:n_cyc] for g in all_grasp])     # (N, n_cyc)
    rarr = np.array([r[:n_cyc] for g in all_release])
    print(f"\n  各次闭爪帧号 (均值 ± 标准差, 极差):")
    for k in range(n_cyc):
        col = garr[:, k]
        print(f"    第{k + 1}次: {col.mean():7.1f} ± {col.std():4.1f}   范围 {col.min()}~{col.max()}"
              f"   {'一致' if col.max() - col.min() <= 1 else '>>> 有差异 <<<'}")
    print(f"\n  各次开爪帧号:")
    for k in range(min(n_cyc, len(rarr[0]))):
        col = rarr[:, k]
        print(f"    第{k + 1}次: {col.mean():7.1f} ± {col.std():4.1f}   范围 {col.min()}~{col.max()}")

    # ---------- 2. 三种定义 ----------
    print()
    print("=" * 78)
    print("2. 三种距离定义对比 (用第 0 集, 与全集的汇总统计)")
    print("=" * 78)

    def build_defs(ee: np.ndarray, grasp: list[int], release: list[int]) -> dict[str, np.ndarray]:
        n = len(ee)
        k_cyc = min(len(grasp), len(release))
        refs = np.stack([ee[f] for f in grasp[:k_cyc]])           # (K,3) 抓取点
        box = ee[release[:k_cyc]].mean(axis=0)                    # (3,)   盒子落点

        # --- A) min_k: 现状 ---
        dA = np.linalg.norm(ee[:, None, :] - refs[None, :, :], axis=2).min(axis=1)

        # --- B) 周期版: 到"本周期对应的那把工具" ---
        # 周期 k 覆盖 [release_{k-1}, release_k)，第1周期从帧0开始。
        # 注意末周期要一直铺到片尾(else n)，否则最后一帧之后会回落到周期0，
        # 在片尾造出一个假的巨大跳变。
        bounds = [0] + [int(f) for f in release[:k_cyc]]
        cyc_of = np.full(n, k_cyc - 1, dtype=np.int64)      # 默认落在最后一个周期
        for k in range(k_cyc):
            lo = bounds[k]
            hi = bounds[k + 1] if k + 1 < len(bounds) else n
            cyc_of[lo:hi] = k
        dB = np.linalg.norm(ee - refs[cyc_of], axis=1)

        # --- C) 目标点集合: 工具抓取点 ∪ 盒子落点, 再取 min ---
        goals = np.concatenate([refs, box[None, :]], axis=0)      # (K+1,3)
        dC = np.linalg.norm(ee[:, None, :] - goals[None, :, :], axis=2).min(axis=1)

        return {"A_min_k": dA, "B_cycle": dB, "C_goals": dC}

    s0 = load_episode(0)
    g0, r0 = find_cycles(s0)
    ee0 = s0[:, EE_SLICE]
    defs0 = build_defs(ee0, g0, r0)

    print(f"  第 0 集: {len(s0)} 帧, 闭爪 {len(g0)} 次, 开爪 {len(r0)} 次")
    print()
    print(f"  {'帧':>6} | {'A min_k':>9} {'B 周期':>9} {'C 目标集':>9} | 阶段")
    print("  " + "-" * 60)
    marks = [(0, "起点"), (60, "接近工具1"), (114, "闭爪1"), (250, "搬运中"),
             (364, "开爪1(盒子)"), (400, "起飞"), (460, "接近工具2"), (514, "闭爪2")]
    for f, nm in marks:
        print(f"  {f:>6} | {defs0['A_min_k'][f]:>9.4f} {defs0['B_cycle'][f]:>9.4f} "
              f"{defs0['C_goals'][f]:>9.4f} | {nm}")

    # ---------- 3. 跳变分析 ----------
    print()
    print("=" * 78)
    print("3. 相邻帧跳变 (|d[f+1]-d[f]|) —— 越小越平滑, 流匹配越好学")
    print("=" * 78)
    for name in ("A_min_k", "B_cycle", "C_goals"):
        d = defs0[name]
        jump = np.abs(np.diff(d))
        print(f"  {name:>8}: 均值 {jump.mean():.5f}  中位 {np.median(jump):.5f}  "
              f"p99 {np.percentile(jump, 99):.4f}  最大 {jump.max():.4f}  (最大跳变在帧 {int(jump.argmax())})")

    # ---------- 4. 取值分布 ----------
    print()
    print("=" * 78)
    print("4. 取值分布 (第 0 集)")
    print("=" * 78)
    for name in ("A_min_k", "B_cycle", "C_goals"):
        d = defs0[name]
        near = (d < 0.05).mean() * 100
        far = (d > 0.20).mean() * 100
        print(f"  {name:>8}: 范围 [{d.min():.4f}, {d.max():.4f}]  "
              f"d<0.05 占 {near:5.1f}%   d>0.20 占 {far:5.1f}%")

    # ---------- 5. B 定义的零交叉点 ----------
    print()
    print("=" * 78)
    print("5. B 定义在周期切换处是否连续")
    print("=" * 78)
    dB = defs0["B_cycle"]
    print(f"  {'切换帧':>8} {'切换前 d':>10} {'切换后 d':>10} {'跳变':>9}")
    for f in r0[:-1]:
        print(f"  {f:>8} {dB[f - 1]:>10.4f} {dB[f]:>10.4f} {dB[f] - dB[f - 1]:>+9.4f}")


if __name__ == "__main__":
    main()
