#!/usr/bin/env python
"""把 clear_eval 转储里**每块的完整 20 步**摊开：量这块预测的时间/距离跨度，并判定
**它到底朝谁收敛**。

## 为什么不能只看 analyze_chunks.py 的"瞄 X(距Ncm)"

那个是 `argmin_j ‖chunk[19,7:10] − ref_j‖` —— "**终点**离谁近"。终点近 ≠ 朝它飞
（绕过去的、停在旁边的，终点都近）。本脚本给三个独立读数：

  A 终点最近  = 老指标，argmin_j ‖p[19]−ref_j‖
  B 朝向最合  = max_j cos( (p[19]−p[0]) , (ref_j−p[0]) )    "整体往哪飞"
  C 逼近最多  = min_j ( d_j[19] − d_j[0] ),  d_j[k]=‖p[k]−ref_j‖
                **字面意义的"朝谁收敛"**：整块走完，到谁的距离减得最多

**默认全在水平面(x,y)上算**：工具都摊在桌面上，模型又常把末端抬到 z≈0.40（比工具高
25cm），三维距离被 z 主导，一移动 15cm 的 y 只让三维距离变几毫米 ⇒ C 会被淹掉。
`--dim3` 切回三维做对照。**B/C 对 A 是增量信息**：若三口径一致，结论才硬。

## 参照点用**场景实测**，不是 goals 表

`tool_refs_d22.json` 的 goals 是**采集期的抓取点**。实测本转储 t=0 时两者差 33~77mm
（手电筒最差 76.6mm），拿 goals 当参照会把"实际就在那儿"读成"偏了"。默认用每帧实测的
`tools_world` 换算到 base 系：

    ref_tool_base = tool_world − base_world − RES_MM/1000

`RES_MM` 是实测固定连杆偏移（`ee_world−base_world−state[7:10]` 均值；残差 std 仅 3~4mm、
z 维 0.08mm ⇒ base 系是纯平移，可以直接这么换）。`--refs table` 切回 goals 表做对照。

## 另外两个读数

  · 起跳差 = ‖chunk[0,7:10] − state[7:10]‖。**规划起点与实际手位的错位**：
    正常闭环重规划该是毫米级（上一块刚执行完）；偏大 = 模型在"假设自己在别处"。
  · 逐帧步长 = ‖p[k+1]−p[k]‖（每步 0.05s），看这块是匀速走还是猛地一窜。

用法:
    analyze_chunk_geometry.py <dump.jsonl> [--refs live|table] [--dim3] [--only-moves]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REFS = os.path.join(HERE, "tool_refs_d22.json")
TRAIN_FPS = 20.0
ENV_DT = 0.005
RES_MM = np.array([-1.81, -6.86, -0.21])   # 实测固定连杆偏移，见 docstring
TOOL_NAMES5 = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
BOX_NAME = "盒子"


def load_table(path: str):
    with open(path) as f:
        d = json.load(f)
    return np.asarray(d["goals"], dtype=np.float64), TOOL_NAMES5 + [BOX_NAME]


def refs_live(rec: dict, box_base: np.ndarray):
    base = np.asarray(rec["base_world"], dtype=np.float64)
    tw = np.asarray(rec["tools_world"], dtype=np.float64) - base - RES_MM / 1000.0
    return np.vstack([tw, box_base[None, :]]), list(rec["tool_names"]) + [BOX_NAME]


def analyze(rec: dict, refs: np.ndarray, names: list[str], dim3: bool) -> dict:
    a = np.asarray(rec["chunk"], dtype=np.float64)           # (20, 18)
    P = a[:, 7:10]                                           # 右臂末端规划位置(base 系)
    cur = np.asarray(rec["state"], dtype=np.float64)[7:10]
    p = P if dim3 else P[:, :2]                              # 三维 或 水平面
    r = refs if dim3 else refs[:, :2]

    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    net = p[-1] - p[0]
    net_n = float(np.linalg.norm(net))

    d = np.linalg.norm(r[None, :, :] - p[:, None, :], axis=2)     # (20, n_ref)
    delta = d[-1] - d[0]
    j_conv, j_end = int(np.argmin(delta)), int(np.argmin(d[-1]))

    if net_n > 1e-9:
        u = net / net_n
        v = r - p[0]
        cos = (v @ u) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        j_head, cos_head = int(np.argmax(cos)), float(np.max(cos))
    else:
        j_head, cos_head = -1, float("nan")

    # 这一块里手离最近工具能压到多近（水平），以及压到最近的工具是谁
    dt = d[:, :5]
    k_best, j_best = np.unravel_index(int(np.argmin(dt)), dt.shape)
    return dict(
        trial=rec.get("trial"), query=rec.get("query"), t=rec.get("t"),
        jump_mm=float(np.linalg.norm(P[0] - cur)) * 1000.0,
        step_med_mm=float(np.median(step)) * 1000.0, step_max_mm=float(step.max()) * 1000.0,
        net_mm=net_n * 1000.0, path_mm=float(step.sum()) * 1000.0,
        span_s=int(rec.get("action_repeat", 10)) * ENV_DT * len(a), span_frames=len(a),
        j_conv=j_conv, d_conv_delta=float(delta[j_conv]) * 1000.0,
        d_conv0_mm=float(d[0, j_conv]) * 1000.0, d_conv19_mm=float(d[-1, j_conv]) * 1000.0,
        j_end=j_end, d_end_mm=float(d[-1, j_end]) * 1000.0,
        j_head=j_head, cos_head=cos_head,
        z0=float(P[0, 2]), z19=float(P[-1, 2]),
        near_tool=names[j_best], near_mm=float(dt[k_best, j_best]) * 1000.0,
        near_k=int(k_best), near_q=int(rec.get("query", -1)),
        grip0=float(a[0, 16]), grip19=float(a[-1, 16]),
        grip_min=float(a[:, 16].min()), grip_max=float(a[:, 16].max()),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--refs", default="live", choices=("live", "table"))
    ap.add_argument("--dim3", action="store_true", help="用三维距离（默认水平面）")
    ap.add_argument("--only-moves", action="store_true", help="只打水平净位移>20mm 的块")
    ap.add_argument("--trace", default="",
                    help="盯一把工具（如 手电筒）：逐块打出它 20 步的距离轨迹 min→max")
    a = ap.parse_args()

    table, names_t = load_table(DEFAULT_REFS)
    with open(a.dump) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if not recs:
        print("转储是空的")
        return 2
    if a.refs == "live":
        refs, names = refs_live(recs[0], table[-1])
        src = "场景实测 tools_world 换算到 base 系"
    else:
        refs, names = table, names_t
        src = "tool_refs_d22.json 的 goals 表（采集期抓取点）"

    rows = [analyze(r, refs, names, a.dim3) for r in recs]
    r0 = rows[0]
    pl = "三维" if a.dim3 else "水平面(x,y)"
    print(f"转储 {a.dump}\n块数 {len(rows)}   参照点: {src}   距离度量: {pl}")
    for n_, q in zip(names, refs):
        print(f"    {n_:<10} ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f})m")
    print(f"\n时间跨度（代码定义，非估计）: {r0['span_frames']} 步 × action_repeat("
          f"{recs[0].get('action_repeat')}) × env_dt({ENV_DT}) = **{r0['span_s']:.2f} s 仿真**"
          f" = {r0['span_frames']} 个训练帧 @{TRAIN_FPS:.0f}fps")
    print(f"执行模式 {recs[0].get('exec_mode')}：整块走完才重问 ⇒ 重规划 {1.0 / r0['span_s']:.1f} Hz"
          f"（每 1.00s 才有一次纠偏机会）\n")

    hdr = (f"{'tr':>2} {'q':>3} {'t(s)':>6} | {'水平净位移':>9} {'路径':>7} {'步长中位':>7} "
           f"{'起跳差':>7} {'z起→末':>13} | {'C 朝谁收敛':<10} {'Δd':>8} | {'B 朝向':<10} "
           f"{'cos':>5} | {'A 终点':<10} {'距':>7} | {'块内最近工具':<10} {'距/第几步':>11} | {'爪':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if a.only_moves and r["net_mm"] < 20:
            continue
        hn = names[r["j_head"]] if r["j_head"] >= 0 else "—"
        print(f"{r['trial']:>2} {r['query']:>3} {r['t']:>6.1f} | {r['net_mm']:>8.1f}mm "
              f"{r['path_mm']:>6.1f}mm {r['step_med_mm']:>6.2f}mm {r['jump_mm']:>6.1f}mm "
              f"{r['z0']:>6.3f}→{r['z19']:.3f} | {names[r['j_conv']]:<10} "
              f"{r['d_conv_delta']:>+7.1f}mm | {hn:<10} {r['cos_head']:>+5.2f} | "
              f"{names[r['j_end']]:<10} {r['d_end_mm']:>6.1f}mm | {r['near_tool']:<10} "
              f"{r['near_mm']:>6.1f}mm/k={r['near_k']:<3} | {r['grip0']:.2f}→{r['grip19']:.2f}")

    # ---------------- 距离跨度 ----------------
    print("\n=== 距离跨度：每块 1.00s 内模型规划自己的手走多远 ===")
    net = np.array([r["net_mm"] for r in rows])
    path = np.array([r["path_mm"] for r in rows])
    jmp = np.array([r["jump_mm"] for r in rows])
    for lbl, v in (("水平净位移", net), ("路径长 Σ|Δp|", path), ("起跳差", jmp)):
        print(f"  {lbl:<14} 中位 {np.median(v):7.1f}mm  均值 {v.mean():7.1f}mm  "
              f"最大 {v.max():7.1f}mm  最小 {v.min():7.1f}mm")
    print(f"  折算速度(净位移/1s) 中位 {np.median(net) / 10:5.1f} cm/s  最大 {net.max() / 10:5.1f} cm/s")
    print(f"  起跳差>50mm 的块: {int((jmp > 50).sum())}/{len(jmp)}"
          f"   ← 规划起点与当时实际手位错位，闭环在打架")

    # ---------------- 三口径对比 ----------------
    print("\n=== 同一批块，三个口径各判给谁（口径不同结论可能不同）===")
    for lbl, key in (("A 终点最近", "j_end"), ("B 朝向最合", "j_head"), ("C 逼近最多", "j_conv")):
        cnt: dict[str, int] = {}
        for r in rows:
            if r[key] >= 0:
                cnt[names[r[key]]] = cnt.get(names[r[key]], 0) + 1
        s = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
        print(f"  {lbl}: {s}   (共 {sum(cnt.values())})")
    allsame = sum(1 for r in rows if r["j_end"] == r["j_head"] == r["j_conv"])
    print(f"  三口径完全一致的块: {allsame}/{len(rows)}"
          f"  ← 一致才说明'朝谁去'这件事没有歧义")

    # ---------------- 逐工具专项 ----------------
    print("\n=== 逐工具专项（C 口径：整块走完离它更近的那些块）===")
    for j, nm in enumerate(names[:5]):
        sel = [r for r in rows if r["j_conv"] == j]
        if not sel:
            print(f"  {nm:<10} 0 块")
            continue
        dd = np.array([r["d_conv_delta"] for r in sel])
        mv = [r for r in sel if r["net_mm"] > 20]
        con = [r for r in sel if r["j_end"] == j and r["j_head"] == j]
        print(f"  {nm:<10} {len(sel):>2} 块  Δd 中位 {np.median(dd):+7.1f}mm "
              f"最接近 {dd.min():+7.1f}mm | 其中真移动(>20mm) {len(mv):>2} 块 | "
              f"A/B/C 全一致 {len(con):>2} 块")

    # ---------------- 块内可达性 ----------------
    print("\n=== 块内可达性：这 1.00s 里手能不能压到某把工具头上 ===")
    near = np.array([r["near_mm"] for r in rows])
    for thr in (20, 50, 100):
        print(f"  块内最近工具距 ≤{thr:>3}mm 的块: {int((near <= thr).sum()):>2}/{len(near)}")
    hit = [r for r in rows if r["near_mm"] <= 50]
    if hit:
        print(f"  这些块瞄的是: " + "  ".join(
            f"{k}:{sum(1 for r in hit if r['near_tool'] == k)}" for k in names[:5]))
    print(f"  块内最近距离中位 {np.median(near):.1f}mm")

    # ---------------- 单工具距离轨迹 ----------------
    if a.trace:
        if a.trace not in names:
            print(f"\n✗ 没有叫 {a.trace} 的参照点；可选 {names}")
            return 2
        j = names.index(a.trace)
        print(f"\n=== 「{a.trace}」逐块距离轨迹（水平面，参照点固定 = 场景 t=0 位置）===")
        print(f"{'q':>3} {'t(s)':>6} | {'d[0]':>7} {'d[19]':>7} {'块内min':>8} {'@k':>3} "
              f"{'Δ(19-0)':>9} | 收敛? | 爪")
        for r, rec in zip(rows, recs):
            P = np.asarray(rec["chunk"], dtype=np.float64)[:, 7:10]
            p = P if a.dim3 else P[:, :2]
            q_ = refs[j] if a.dim3 else refs[j, :2]
            dd = np.linalg.norm(p - q_, axis=1) * 1000.0
            k = int(np.argmin(dd))
            flag = "↓收敛" if dd[-1] < dd[0] - 1 else ("↑远离" if dd[-1] > dd[0] + 1 else "→持平")
            print(f"{r['query']:>3} {r['t']:>6.1f} | {dd[0]:>6.1f}mm {dd[-1]:>6.1f}mm "
                  f"{dd.min():>7.1f}mm {k:>3} {dd[-1] - dd[0]:>+8.1f}mm | {flag} | "
                  f"{r['grip0']:.2f}→{r['grip19']:.2f}")
        dc = [(r["t"], r["query"]) for r, rec in zip(rows, recs)
              if np.asarray(rec["chunk"])[0, 16] < 0.3 and np.asarray(rec["chunk"])[-1, 16] > 0.5]
        print(f"  这 81 块里，{a.trace} 的爪子从张开跨到闭合的块: "
              f"{[(f't={t:.1f}s', f'q={q}') for t, q in dc] or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
