#!/usr/bin/env python3
"""分析 --dump_chunks 的转储，回答"模型在停顿期间还在不在推理"。

背景（2026-09-20）：去掉挂钟时限后，idxR 演练跑到 292s，中间出现两段长时间不动
（t≈132-175 共 43s、t≈211-292 共 81s）。用户问："停止之后模型是否还在参与推理？
按道理只要有输入就会输出动作序列。"  这个脚本用转储里的原始数据回答它，不靠推测。

三个量必须分开看，它们指向完全不同的结论：
  1. **state 是否冻结**  —— 冻结 ⇒ 模型每轮收到完全相同的输入 ⇒ 输出必然相同 ⇒ 定点。
  2. **action 是否为零** —— 非零但不动 ⇒ 不是"模型躺平"，是**下发后被抵消/顶住**。
  3. **action 是否变化** —— 每轮 chunk 都不一样 ⇒ 推理确实在跑，不是进程卡死。

判据（写死在代码里，看数之前就定好）：
  · 进程卡死    : step 不涨（本脚本查 step 单调递增）
  · 定点        : state 逐维 |Δ| < 1e-6 且 action 逐维 |Δ| < 1e-4
  · 零输出      : |action[7:14]| 的 max < 0.01（右臂全零）
  · 抵消/顶住    : action 非零且变化，但 |ΔEE| < 1mm/chunk
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# d22 布局（与 runner / 训练一致）
EE = slice(7, 10)        # 右臂末端位置（base 系）
RQUAT = slice(10, 14)    # 右臂末端四元数
RARM_A = slice(7, 14)    # 动作里的右臂 7 维
RGRIP_A = slice(16, 18)  # 动作里的右夹爪 2 维
GOAL_D, GOAL_Y, CYC_D, CYC_I = 18, 19, 20, 21


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/home/dgl/lxc/.runlogs/rollout_idxR_notl.jsonl")
    rows = load(p)
    print(f"档 {p.name}：{len(rows)} 个 chunk")

    st = np.asarray([r["state"] for r in rows], dtype=np.float64)
    ac = np.asarray([r["chunk"] for r in rows], dtype=np.float64)   # (N,20,18)
    t = np.asarray([r["t"] for r in rows], dtype=np.float64)
    step = np.asarray([r["step"] for r in rows], dtype=np.int64)
    cyc = np.asarray([r["cycle_idx"] for r in rows], dtype=np.int64)

    # ---- 判据 0：进程有没有卡死（step 必须严格递增）----
    dstep = np.diff(step)
    print(f"\n[进程] step 单调递增: {'✓ 是' if (dstep > 0).all() else '✗ 否'}"
          f"  每 chunk 步数 min/中位/max = {dstep.min()}/{int(np.median(dstep))}/{dstep.max()}")
    print(f"       挂钟/chunk 中位 {np.median(np.diff(t)):.2f}s"
          f"  ⇒ 全程 {t[-1]:.0f}s / {step[-1]} 步")

    # ---- 每 chunk 的 state 位移与动作强度 ----
    d_ee = np.linalg.norm(np.diff(st[:, EE], axis=0), axis=1) * 1000.0     # mm
    d_st = np.abs(np.diff(st, axis=0)).max(axis=1)                        # 逐维最大 |Δ|
    a_arm = np.abs(ac[:, :, RARM_A]).max(axis=(1, 2))                     # 右臂动作幅度
    a_grip = ac[:, :, RGRIP_A].mean(axis=(1, 2))                          # 右夹爪目标均值
    d_ac = np.abs(np.diff(ac, axis=0)).max(axis=(1, 2))                   # 相邻 chunk 动作差
    # 动作里"右臂目标"与"当前 state 位置"的差距（= 模型想让它走多远）
    a_res = np.linalg.norm(ac[:, -1, EE] - st[:, EE], axis=1) * 1000.0    # 末帧 mm

    STALL_MM = 1.0
    stall = d_ee < STALL_MM
    print(f"\n[停顿] 判据 |ΔEE| < {STALL_MM} mm/chunk")
    print(f"       满足的 chunk 数 {stall.sum()}/{len(stall)}"
          f"（占 {100*stall.mean():.0f}%）")

    # 连续段
    segs = []
    i = 0
    while i < len(stall):
        if stall[i]:
            j = i
            while j + 1 < len(stall) and stall[j + 1]:
                j += 1
            if t[j] - t[i] >= 5.0:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    print(f"       ≥5s 的连续停顿段 {len(segs)} 段：")
    for i, j in segs:
        print(f"         t={t[i]:6.1f}→{t[j]:6.1f}s ({t[j]-t[i]:5.1f}s)"
              f"  idx={cyc[i]}  步 {step[i]}→{step[j]}"
              f"  位移 {(np.linalg.norm(st[j, EE]-st[i, EE]))*1000:5.1f} mm")
    if not segs:
        print("         （无）")

    # ---- 判据：停在段里，state 冻没冻、action 零没零、action 变没变 ----
    print("\n[停顿段的三个判据]  （冻结 / 零输出 / 抵消）")
    hdr = (f"  {'t范围':>16s} {'|Δstate|max':>12s} {'|Δact|max':>10s}"
           f" {'臂act最大':>10s} {'爪目标':>7s} {'末帧残差mm':>10s}  结论")
    print(hdr)
    for i, j in segs:
        s_d = d_st[i:j].max() if j > i else 0.0
        a_d = d_ac[i:j].max() if j > i else 0.0
        a_m = a_arm[i:j + 1].max()
        g_m = a_grip[i:j + 1].mean()
        r_m = a_res[i:j + 1].max()
        if s_d < 1e-6 and a_d < 1e-4:
            verdict = "定点：输入输出全冻 ⇒ 模型认为已完成"
        elif a_m < 0.01:
            verdict = "零输出：右臂动作恒为零"
        elif a_d < 1e-4:
            verdict = "恒定非零：模型在推但被顶住/抵消"
        else:
            verdict = "动作每轮在变：推理在跑，但推不动"
        print(f"  {t[i]:6.1f}→{t[j]:6.1f}s {s_d:12.2e} {a_d:10.2e}"
              f" {a_m:10.4f} {g_m:7.3f} {r_m:10.1f}  {verdict}")

    # ---- cycle 推进点：当时 EE 离各工具多远、入箱几件 ----
    print("\n[cycle_idx 推进点]")
    chg = np.flatnonzero(np.diff(cyc) != 0) + 1
    for k in chg:
        print(f"  t={t[k]:6.1f}s  step={step[k]:6d}  idx {cyc[k-1]} → {cyc[k]}"
              f"  goal_y {st[k-1, GOAL_Y]:+.4f} → {st[k, GOAL_Y]:+.4f}"
              f"  goal_dist {st[k, GOAL_D]:.3f}  EE={np.round(st[k, EE], 3).tolist()}")
    print(f"  idx 终值 {cyc[-1]}（<=5 表示五件没收完）")

    # ---- 全段 EE 轨迹摘要：每 20s 一行 ----
    print("\n[EE 轨迹] 每 ~20s 采样")
    print(f"  {'t':>7s} {'idx':>4s} {'EE_x':>7s} {'EE_y':>7s} {'EE_z':>7s}"
          f" {'爪state':>8s} {'goal_dist':>10s} {'|ΔEE|/chunk mm':>15s}")
    last = -1e9
    for k in range(len(rows)):
        if t[k] - last >= 20.0 or k == len(rows) - 1:
            last = t[k]
            dd = d_ee[k - 1] if k > 0 else float("nan")
            print(f"  {t[k]:7.1f} {cyc[k]:4d} {st[k,7]:7.3f} {st[k,8]:7.3f}"
                  f" {st[k,9]:7.3f} {st[k,16]:8.4f} {st[k,GOAL_D]:10.3f} {dd:15.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
