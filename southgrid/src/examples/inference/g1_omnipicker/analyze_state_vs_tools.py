"""对读：训练数据里的 state↔工具关系 vs ④ 演练里实际发生的。

要回答的问题（用户原话）："为什么你已知最左边的工具位置，已知要去抓取这个工具，
结果最后你还是不去，直接略过去了？"

思路：state 里真正携带"该抓哪一件"的是第 20/21 维（cycle_dist 是到当前轮目标的距离，
goal_y 是当前轮目标的 y）。而 tool_refs 是有序的 —— 按 y 从 -0.0845 到 -0.4925 排开，
正好是"左→右"的 5 件。所以：

  A. 先看**训练数据**里，idx=k 的那段时间末端到底在谁头上。
     如果 idx=k 期间末端确实黏在第 k-1 个参照点上，那 tool_refs 的顺序 = 抓取顺序，
     而且 dim21 就是"轮到第几件"的唯一载体。
  B. 再看**④ 演练 dump**里同一组量，找出它从哪一刻开始偏离、
     以及那一刻是什么把它顶掉的。

两边都用同一套参照点、同一个坐标系（右臂末端 base 系），所以可直接比。

用法：/home/dgl/lxc/openpi/.venv/bin/python analyze_state_vs_tools.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
REF_JSON = os.path.join(base_dir, "tool_refs_d22.json")


def load_refs():
    with open(REF_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)
    tools = np.asarray(j["tool_refs"], dtype=np.float64)      # 5 件，已按抓取顺序
    box = np.asarray(j["box_ref"], dtype=np.float64)
    goals = np.asarray(j["goals"], dtype=np.float64)          # 5 工具 + 盒子
    return tools, box, goals


def nearest_refs(p, tools, box):
    """返回 (离最近的工具下标, 该距离, 离盒子的距离, 到 6 个 goal 的 min)。"""
    dt = np.linalg.norm(tools - p, axis=1)
    k = int(dt.argmin())
    db = float(np.linalg.norm(box - p))
    return k, float(dt[k]), db, float(min(dt.min(), db))


# --------------------------------------------------------------------------
# A. 训练数据
# --------------------------------------------------------------------------
def part_a(ds, episode, tools, box):
    import pandas as pd
    fs = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[episode])["observation.state"].values)
    print(f"\n{'='*100}\nA. 训练数据 {os.path.basename(ds)} 第{episode}集  {st.shape[0]} 帧\n{'='*100}")
    if st.shape[1] < 22:
        print("  (该数据集没有 cycle_idx，跳过)")
        return

    idx = np.rint(st[:, 21]).astype(int)
    # idx 的变化点
    ch = np.flatnonzero(np.diff(idx) != 0) + 1
    print(f"  cycle_idx 变化帧: {ch.tolist()}   (共 {len(ch)} 次)")
    print(f"  cycle_idx 取值序列: {sorted(set(idx.tolist()))}")

    print(f"\n  {'idx':>4s} {'帧数':>6s} {'帧区间':>14s} {'末端base位置(区间均值)':>32s} "
          f"{'最近工具':>9s} {'距最近工具':>10s} {'末端到它最短':>12s} {'到盒子最短':>10s} {'末端y均值':>10s}")
    for k in sorted(set(idx.tolist())):
        m = np.flatnonzero(idx == k)
        if m.size == 0:
            continue
        ee = st[m, 7:10]
        mean_ee = ee.mean(axis=0)
        # 这段里末端离 5 个参照点各自的最短距离
        d = np.linalg.norm(ee[:, None, :] - tools[None, :, :], axis=2)   # (n,5)
        per_tool = d.min(axis=0)
        kbest = int(per_tool.argmin())
        db = np.linalg.norm(ee - box, axis=1).min()
        name = f"#{kbest}({_tname(kbest)})"
        print(f"  {k:>4d} {m.size:>6d} {m[0]:>6d}..{m[-1]:<6d} "
              f"{str(np.round(mean_ee,4).tolist()):>32s} {name:>9s} "
              f"{per_tool[kbest]*1000:>7.1f} mm {per_tool[kbest]*1000:>9.1f} mm "
              f"{db*1000:>7.1f} mm {ee[:,1].mean():>+10.4f}")

    ee0 = st[0, 7:10]
    print(f"\n  帧0: 末端 {np.round(ee0,4).tolist()}  goal_y={st[0,19]:+.4f} "
          f"cycle_dist={st[0,20]:.4f}  cycle_idx={idx[0]}")
    k, dk, db, dmin = nearest_refs(ee0, tools, box)
    print(f"       离最近的工具 = #{k}({_tname(k)}) {dk*1000:.1f} mm ；离盒子 {db*1000:.1f} mm")
    print(f"       ⚠ dim18(goal_dist)={st[0,18]:.4f} 而 dim20(cycle_dist)={st[0,20]:.4f} —— "
          f"goal_dist 是到 6 个 goal 的 **min**（这里是 #{int(np.argmin(np.linalg.norm(np.vstack([tools,box])-ee0,axis=1)))}），"
          f"cycle_dist 才是到**本轮目标**的")

    # 末端 y 是否单调左→右
    print(f"\n  各 idx 期间末端 y 均值（应当逐步变负 = 从左到右）：")
    ys = [(k, st[np.flatnonzero(idx == k), 7:10][:, 1].mean()) for k in sorted(set(idx.tolist()))]
    print("    " + "  ".join(f"idx{k}:{v:+.3f}" for k, v in ys))
    mono = all(ys[i][1] > ys[i+1][1] for i in range(len(ys)-1))
    print(f"    单调递减(左→右) = {mono}")


def _tname(k):
    return ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"][k] if 0 <= k < 5 else "?"


# --------------------------------------------------------------------------
# B. ④ 演练 dump
# --------------------------------------------------------------------------
def part_b(path, tools, box):
    print(f"\n{'='*100}\nB. ④ 演练 dump {os.path.basename(path)}\n{'='*100}")
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"  {len(recs)} 条推理记录")
    if not recs:
        return
    print(f"  字段: {sorted(recs[0].keys())}")
    has_cyc = "cycle_idx" in recs[0]
    print(f"  含 cycle_idx 字段 = {has_cyc}（旧 dump 没有，从 state[21] 里取）\n")

    def get(r, key, i=None):
        if key in r and r[key] is not None:
            v = r[key]
            return float(v[i]) if i is not None else float(v)
        return np.nan

    print(f"  {'t(s)':>7s} {'idx':>4s} {'goal_y':>9s} {'cyc_d':>7s} "
          f"{'末端y':>8s} {'命令目标y':>10s} {'命令到最近工具':>14s} {'命令瞄向':>12s} {'夹爪闭合件数':>10s}")
    prev_idx = None
    for i, r in enumerate(recs):
        st = np.asarray(r["state"], dtype=np.float64)
        cyc = int(round(st[21])) if st.size >= 22 else int(get(r, "cycle_idx"))
        gy = st[19] if st.size >= 22 else np.nan
        cd = st[20] if st.size >= 22 else np.nan
        ee = st[7:10]
        ch = np.asarray(r["chunk"], dtype=np.float64)   # (20,18)
        tgt = ch[:, 7:10]                                # 命令的右臂末端目标(base 系)
        # 该 chunk 内命令目标离 5 个工具参照点各自的最短距离
        d = np.linalg.norm(tgt[:, None, :] - tools[None, :, :], axis=2)
        per_tool = d.min(axis=0)
        kb = int(per_tool.argmin())
        # 夹爪：chunk 里第16维是 r_grip_inner_norm，>0.5 视为闭合
        nclose = int((ch[:, 16] > 0.5).sum())
        mark = ""
        if prev_idx is not None and cyc != prev_idx:
            mark = f"   <== idx {prev_idx}->{cyc}"
            prev_idx = cyc
        elif prev_idx is None:
            prev_idx = cyc
        # 只在 idx 变化、或每 20 条打一次，避免刷屏
        if mark or i % 20 == 0 or i == len(recs) - 1:
            print(f"  {get(r,'t'):>7.2f} {cyc:>4d} {gy:>+9.4f} {cd:>7.3f} "
                  f"{ee[1]:>+8.4f} {tgt[:,1].mean():>+10.4f} "
                  f"{per_tool[kb]*1000:>11.1f} mm {('#'+str(kb)+_tname(kb)):>12s} "
                  f"{nclose:>10d}{mark}")

    # idx 变化点的细节
    print(f"\n  --- idx 变化点细节 ---")
    prev = None
    for i, r in enumerate(recs):
        st = np.asarray(r["state"], dtype=np.float64)
        cyc = int(round(st[21])) if st.size >= 22 else int(get(r, "cycle_idx"))
        if prev is not None and cyc != prev:
            tw = r.get("tools_world") or []
            t_pos = [np.asarray(t, dtype=np.float64) for t in tw if t]
            st_pos = [np.asarray(t, dtype=np.float64) for t in (r.get("state") or [])]
            ee = st[7:10]
            d = np.linalg.norm(tools - ee, axis=1)
            elems = "  ".join(
                f"#{j}{_tname(j)}:{d[j]*1000:.0f}mm" for j in range(5))
            print(f"    t={get(r,'t'):6.2f}s  idx {prev}->{cyc}  末端离各工具: {elems}")
            prev = cyc
        elif prev is None:
            prev = cyc

    # 工具是否移动过（用 tools_world 或 state 里的 tool_world）
    print(f"\n  --- 各工具位移（首末对比） ---")
    k0 = next((k for k, r in enumerate(recs) if r.get("tools_world")), None)
    if k0 is not None and recs[-1].get("tools_world"):
        a = [np.asarray(t, dtype=np.float64) if t else None for t in recs[k0]["tools_world"]]
        b = [np.asarray(t, dtype=np.float64) if t else None for t in recs[-1]["tools_world"]]
        for j in range(min(len(a), len(b))):
            if a[j] is None or b[j] is None:
                print(f"    #{j}{_tname(j)}: 缺数据")
                continue
            mv = float(np.linalg.norm(b[j] - a[j]))
            dz = float(b[j][2] - a[j][2])
            print(f"    #{j}{_tname(j)}: 位移 {mv*1000:7.1f} mm   Δz {dz*1000:+7.1f} mm   "
                  f"{'← 动过' if mv > 0.02 else '（没动）'}")
    else:
        print("    dump 里没有 tools_world 字段（旧格式），跳过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dump", type=str, default="/home/dgl/lxc/.runlogs/d22_no_score_chunks.jsonl")
    args = ap.parse_args()

    tools, box, goals = load_refs()
    print("参照点（tool_refs_d22.json，已按抓取顺序）：")
    for j, t in enumerate(tools):
        print(f"  #{j} {_tname(j):<10s} {np.round(t,4).tolist()}")
    print(f"  盒子                {np.round(box,4).tolist()}")
    print(f"  → 5 件的 y 从 {tools[0][1]:+.4f} 到 {tools[4][1]:+.4f}"
          f"，跨度 {abs(tools[4][1]-tools[0][1])*1000:.0f} mm，"
          f"顺序 = y 递减 = 左→右")

    part_a(args.dataset, args.episode, tools, box)
    if os.path.exists(args.dump):
        part_b(args.dump, tools, box)
    else:
        print(f"\n[跳过 B] 找不到 {args.dump}")


if __name__ == "__main__":
    main()
