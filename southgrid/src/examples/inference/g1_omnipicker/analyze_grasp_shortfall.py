"""单回合演练后：抓取为什么没成 —— 是策略没瞄到位，还是手臂没跟到位？

单回合演练（.runlogs/d22_single_ep.jsonl）里发生了质变：
  · cycle_idx 前 147s 一直稳在 1（旧写法 3.9s 就被顶到 2）→ "掠过最左边"这个症状在
    state 层面**已经修好**。
  · 手臂确实**先去了最左边那把扳手**并在它附近闭合了夹爪（ee_tool 0.192→0.057，grip→+2.00）
    —— 这正是训练里 idx=1 该做的事。
  · 但扳手**一步没动**（位移 0.000m），夹爪电机却顶到 +2.00（完全合拢）= 夹的是空气。
    训练数据里 idx=1 期间末端能贴到参照点 **10.6mm**，这里只到 **57mm**，差 46mm。

所以现在唯一的问题变成：**为什么停在 46mm 外就闭爪了？**
两种可能，看两个数就能分开：
  · 命令的末端目标（chunk 的 7:10 维）本身就没到参照点附近 → **策略**的锅
  · 命令到了，但实际末端跟不上 → **控制器**的锅

用法：/home/dgl/lxc/openpi/.venv/bin/python analyze_grasp_shortfall.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
TNAME = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, default="/home/dgl/lxc/.runlogs/d22_single_ep.jsonl")
    ap.add_argument("--ref", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--start", type=str, default=os.path.join(base_dir, "start_ref_d22.json"))
    args = ap.parse_args()

    with open(args.ref, "r", encoding="utf-8") as f:
        tools = np.asarray(json.load(f)["tool_refs"], dtype=np.float64)
    with open(args.start, "r", encoding="utf-8") as f:
        s0 = np.asarray(json.load(f)["state22"], dtype=np.float64)

    recs = []
    with open(args.dump, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    print(f"[dump] {len(recs)} 条；字段 {sorted(recs[0].keys())}")

    T = np.array([r["t"] for r in recs], dtype=np.float64)
    S = np.stack([np.asarray(r["state"], dtype=np.float64) for r in recs])
    EE = S[:, 7:10]
    CMD = np.stack([np.asarray(r["chunk"], dtype=np.float64)[:, 7:10].mean(axis=0) for r in recs])
    CMDEND = np.stack([np.asarray(r["chunk"], dtype=np.float64)[-1, 7:10] for r in recs])
    GRIP = np.stack([np.asarray(r["chunk"], dtype=np.float64)[:, 16].mean() for r in recs])
    IDX = np.rint(S[:, 21]).astype(int)
    d_ee0 = np.linalg.norm(EE - tools[0], axis=1)
    d_cmd0 = np.linalg.norm(CMD - tools[0], axis=1)
    d_cmde0 = np.linalg.norm(CMDEND - tools[0], axis=1)

    print(f"\n=== 全程：末端/命令 离 **#0 扳手**参照点 的最短距离 ===")
    print(f"  实际末端最近        {d_ee0.min()*1000:7.1f} mm  @ t={T[d_ee0.argmin()]:.1f}s")
    print(f"  命令(chunk20条均值)最近 {d_cmd0.min()*1000:7.1f} mm  @ t={T[d_cmd0.argmin()]:.1f}s")
    print(f"  命令(chunk末条)最近   {d_cmde0.min()*1000:7.1f} mm  @ t={T[d_cmde0.argmin()]:.1f}s")
    print(f"  【训练数据里 idx=1 期间末端能到 10.6mm】")

    # 闭爪边沿
    closed = GRIP > 0.5
    edges = np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1
    opened = np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1
    print(f"\n=== 夹爪闭合边沿（chunk 均值 >0.5）===")
    print(f"  闭合起: {[f'{T[i]:.1f}s' for i in edges[:8]]}")
    print(f"  张开起: {[f'{T[i]:.1f}s' for i in opened[:8]]}")

    if edges.size:
        i = int(edges[0])
        print(f"\n=== 第一次闭合前后明细（t={T[i-2]:.1f}..{T[min(i+3,len(recs)-1)]:.1f}s）===")
        print(f"  {'t':>7s} {'idx':>4s} {'夹爪均值':>8s} {'末端→#0':>10s} {'命令→#0':>10s} "
              f"{'命令末条→#0':>12s} {'末端→各工具最近':>16s}")
        for k in range(max(0, i - 4), min(i + 4, len(recs))):
            dmin = float(np.linalg.norm(tools - EE[k], axis=1).min())
            print(f"  {T[k]:>7.2f} {IDX[k]:>4d} {GRIP[k]:>8.3f} {d_ee0[k]*1000:>7.1f} mm "
                  f"{d_cmd0[k]*1000:>7.1f} mm {d_cmde0[k]*1000:>9.1f} mm {dmin*1000:>13.1f} mm")

    # 工具是否移动
    if recs[0].get("tools_world") and recs[-1].get("tools_world"):
        TW = [r.get("tools_world") for r in recs]
        print(f"\n=== 各工具位移（用 tools_world，世界系）===")
        for j in range(5):
            a = next((np.asarray(w[j], dtype=np.float64) for w in TW if w and w[j]), None)
            b = next((np.asarray(w[j], dtype=np.float64) for w in reversed(TW) if w and w[j]), None)
            mv = np.linalg.norm(b - a) if a is not None and b is not None else float("nan")
            # 逐帧最大位移，看有没有"动过又回来"
            arr = np.array([np.asarray(w[j], dtype=np.float64) for w in TW if w and w[j]])
            mx = float(np.linalg.norm(arr - arr[0], axis=1).max()) if len(arr) else float("nan")
            print(f"  #{j}{TNAME[j]:<10s} 首末位移 {mv*1000:8.1f} mm   全程最大位移 {mx*1000:8.1f} mm")

    # 位置跟踪误差
    err = np.linalg.norm(EE - CMD, axis=1)
    print(f"\n=== 末端对命令的跟踪误差（全程）===")
    print(f"  中位数 {np.median(err)*1000:.1f} mm   90 分位 {np.percentile(err,90)*1000:.1f} mm   "
          f"最大 {err.max()*1000:.1f} mm")

    # 起始状态对比
    print(f"\n=== 起始状态 vs 训练帧0（逐维）===")
    names = ["Lx", "Ly", "Lz", "Lq0", "Lq1", "Lq2", "Lq3",
             "Rx", "Ry", "Rz", "Rq0", "Rq1", "Rq2", "Rq3",
             "Lg0", "Lg1", "Rg0", "Rg1", "goal_d", "goal_y", "cyc_d", "cyc_i"]
    d = np.abs(S[0] - s0)
    for i, n in enumerate(names):
        flag = "   <== 差得多" if d[i] > 0.02 else ""
        print(f"  {i:2d} {n:7s} 演练 {S[0][i]:+.4f}  训练 {s0[i]:+.4f}  差 {d[i]:.4f}{flag}")


if __name__ == "__main__":
    main()
