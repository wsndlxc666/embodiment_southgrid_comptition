#!/usr/bin/env python
"""量"手臂往下掉"：OSC 的指令位姿 vs 实际到位姿。

背景（2026-09-21 查实）：
  · 机械臂 7 个关节在 XML 里**只有 `_mctrl`（裸力矩），没有 `_pctrl`**
    （夹爪/头部/轮子都有 pctrl，带 biastype=affine biasprm="0 -1" 的位置伺服）。
    ⇒ 手臂靠的是 OSC 算力矩，**模型里没有重力补偿**，全靠 kp 这根"弹簧"顶着。
  · 采集器 `g1_omnipicker_collect_clear5_lerobot.py:500`  --kp 默认 **220**
  · 评分 runner `run_task3_tool_storage_scored.py:750`   --kp 默认 **150**
    两边都写进 `controller.kp`（6 维）+ `kd = 2*sqrt(kp)`，且**没有任何启动脚本传过 --kp**
    ⇒ 100 集训练数据在 220 下采，所有评测在 150 下跑。
  · 位置伺服的稳态下垂 ∝ 负载力矩/kp ⇒ 150 下的下垂是 220 下的 **220/150 = 1.467 倍**。
    用户观察"手臂一直往下掉、比训练时重" 的物理量就是这个比值。

本脚本用 rollout 的 chunks jsonl 把下垂直接量出来：
  每条记录 = 一个 chunk（20 步）。chunk[19][7:10] 是这一块**最后一步的指令 EE**（base 系），
  下一条记录的 state[7:10] 就是那一步**实际到达的 EE**。两者之差 = 稳态跟踪误差 = 下垂。
  再按夹爪开/合分开统计：合爪=带载，开爪=空载。带载比空载多出来的那部分就是"工具压的"。

用法：
  /home/dgl/miniconda3/envs/orcalab/bin/python measure_osc_droop.py [jsonl]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

DEFAULT = "/home/dgl/lxc/.runlogs/rollout_idxR11999_grip_chunks.jsonl"
EE = slice(7, 10)
GRIP_R_INNER = 16          # d22: 14/15 左爪, 16/17 右爪
GRIP_CLOSED = 0.5          # 归一化：1.0=闭合, 0.048=张开
KP_TRAIN, KP_EVAL = 220.0, 150.0


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    recs = load(path)
    print(f"读 {os.path.basename(path)}：{len(recs)} 条 chunk 记录")
    if len(recs) < 3:
        print("记录太少，量不了"); return

    # 每条记录：指令终点 = chunk[-1][7:10]，实际 = 下一条的 state[7:10]
    rows = []
    for i in range(len(recs) - 1):
        a = recs[i]
        ch = np.asarray(a["chunk"], dtype=np.float64)
        if ch.ndim != 2 or ch.shape[1] < 18:
            continue
        tgt = ch[-1, EE]                                    # 指令（base 系）
        cur = np.asarray(recs[i]["state"], dtype=np.float64)[EE]
        nxt = np.asarray(recs[i + 1]["state"], dtype=np.float64)[EE]   # 实际到达
        err = nxt - tgt                                     # 实际 - 指令
        grip = float(np.asarray(recs[i + 1]["state"], dtype=np.float64)[GRIP_R_INNER])
        rows.append((i, a["t"], a.get("cycle_idx", -1), grip, err,
                     np.linalg.norm(err) * 1000, err[2] * 1000, nxt, tgt))

    E = np.array([r[4] for r in rows])
    n = np.array([r[5] for r in rows])
    dz = np.array([r[6] for r in rows])
    grip = np.array([r[3] for r in rows])
    closed = grip > GRIP_CLOSED

    print(f"\n=== 全体（{len(rows)} 个 chunk 边界） ===")
    print(f"  |实际-指令| 中位 {np.median(n):6.1f} mm   均值 {n.mean():6.1f}   "
          f"90分位 {np.percentile(n,90):6.1f}")
    print(f"  Δz 中位 {np.median(dz):+6.1f} mm   （负 = 实际比指令**低** = 往下掉）")

    for lab, m in (("合爪（带载）", closed), ("开爪（空载）", ~closed)):
        if m.sum() == 0:
            continue
        print(f"\n=== {lab}：{int(m.sum())} 个 ===")
        print(f"  |实际-指令| 中位 {np.median(n[m]):6.1f} mm   "
              f"Δz 中位 {np.median(dz[m]):+6.1f} mm   "
              f"Δz 均值 {dz[m].mean():+6.1f} mm")
        print(f"  三个轴的有向偏差均值 (Δx,Δy,Δz) = "
              f"({E[m,0].mean()*1000:+.1f}, {E[m,1].mean()*1000:+.1f}, {E[m,2].mean()*1000:+.1f}) mm")

    if closed.sum() and (~closed).sum():
        d_load = np.median(n[closed]) - np.median(n[~closed])
        print(f"\n=== 带载 − 空载 = {d_load:+.1f} mm  ← 这一部分就是「工具压出来的」额外下垂 ===")

    # 按 cycle 分段看（idx 越大 = 拿过的工具越多）
    print("\n=== 按 cycle_idx 分段 ===")
    ci = np.array([r[2] for r in rows])
    for c in sorted(set(ci.tolist())):
        m = ci == c
        if m.sum() < 3:
            continue
        print(f"  idx={c}: n={int(m.sum()):>3}  |err|中位 {np.median(n[m]):6.1f} mm   "
              f"Δz中位 {np.median(dz[m]):+6.1f} mm   合爪占比 {100*closed[m].mean():5.1f}%")

    # 外推：如果按训练的 kp=220 跑，下垂会是多少
    print(f"\n=== 换算到训练刚度 kp={KP_TRAIN:.0f}（当前 {KP_EVAL:.0f}） ===")
    s = KP_EVAL / KP_TRAIN
    print(f"  kp 比值 {s:.3f} ⇒ 评测的下垂是训练时的 {1/s:.3f} 倍")
    print(f"  若实测下垂中位 {np.median(n):.1f} mm，则 kp=220 下约为 "
          f"{np.median(n)*s:.1f} mm（少 {np.median(n)*(1-s):.1f} mm）")
    print(f"  （严格讲只有**负载相关**的那部分按 1/kp 缩；恒定偏置部分不缩，所以这是上界）")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 补充分析（2026-09-21 追加）：上一节把「chunk 末步指令」和「20 步后实际」直接相减，
# 量到的是**跟踪滞后 + 指令自身还在动**的混合量，不是稳态误差。
# 要量"手臂被压下去多少"必须在**指令已经停住**的片段上看残差 —— 那时残差 = 稳态误差。
# ---------------------------------------------------------------------------
def stationary(path):
    recs = load(path)
    rows = []
    for i in range(len(recs) - 1):
        a = recs[i]
        ch = np.asarray(a["chunk"], dtype=np.float64)
        if ch.ndim != 2 or ch.shape[1] < 18:
            continue
        # 指令在最后 5 步内自己动了多少 —— 小才是"停住了"
        plan_move = float(np.linalg.norm(ch[-1, EE] - ch[-6, EE])) * 1000
        tgt = ch[-1, EE]
        s_next = np.asarray(recs[i + 1]["state"], dtype=np.float64)
        nxt = s_next[EE]
        err = nxt - tgt
        rows.append(dict(i=i, t=a["t"], idx=a.get("cycle_idx", -1),
                         plan_move=plan_move, grip=float(s_next[GRIP_R_INNER]),
                         err=err, n=float(np.linalg.norm(err)) * 1000,
                         dz=float(err[2]) * 1000,
                         ee_h=float(nxt[2])))
    return rows


def main2(path):
    rows = stationary(path)
    pm = np.array([r["plan_move"] for r in rows])
    print("\n\n" + "=" * 72)
    print("补充：只看**指令已停住**的片段（末 5 步指令位移 < 1mm）—— 残差=稳态误差")
    print("=" * 72)
    print(f"  全部 {len(rows)} 个边界，末5步指令位移：中位 {np.median(pm):.2f} mm，"
          f"<1mm 的占 {100*np.mean(pm<1):.1f}%")
    sel = [r for r in rows if r["plan_move"] < 1.0]
    if len(sel) < 5:
        print("  停住的样本太少，量不了"); return
    print(f"  停住的 {len(sel)} 个：")
    for lab, m in (("合爪（带载）", [r for r in sel if r["grip"] > GRIP_CLOSED]),
                   ("开爪（空载）", [r for r in sel if r["grip"] <= GRIP_CLOSED])):
        if len(m) < 3:
            print(f"    {lab}: 仅 {len(m)} 个，略"); continue
        n = np.array([r["n"] for r in m]); dz = np.array([r["dz"] for r in m])
        E = np.array([r["err"] for r in m]) * 1000
        print(f"    {lab} n={len(m):>3}: |残差| 中位 {np.median(n):5.1f} mm   "
              f"Δz 中位 {np.median(dz):+6.1f} mm   有向均值 "
              f"({E[:,0].mean():+6.1f},{E[:,1].mean():+6.1f},{E[:,2].mean():+6.1f}) mm")
    # 最干净的静止样本：完整列出
    print("\n  残差最大的 8 个停住片段（|残差| 降序）：")
    print(f"    {'t':>7} {'idx':>4} {'夹爪':>6} {'指令位移':>9} {'|残差|':>8} {'Δz':>8} {'实际z':>8}")
    for r in sorted(sel, key=lambda r: -r["n"])[:8]:
        g = "合" if r["grip"] > GRIP_CLOSED else "开"
        print(f"    {r['t']:>7.1f} {r['idx']:>4} {g:>6} {r['plan_move']:>8.2f}mm "
              f"{r['n']:>7.1f}mm {r['dz']:>+7.1f}mm {r['ee_h']:>7.3f}m")
