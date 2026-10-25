#!/usr/bin/env python3
"""决定性实验：policy 到底有没有在听 prompt？

为什么要做这个：仿真里 v2/9000 跑出 229 次推理，"瞄准 prompt 指定那件工具"只有 13.5%，
比 5 件工具瞎猜的 20% 还低。这有两种完全不同的原因，而表现一模一样：
  (a) prompt 根本没送进模型（键名错/被截断/被覆盖）→ 改接线就能救；
  (b) prompt 进去了，但模型没学会用（容量/数据/训练）→ 改接线没用。
只看仿真分不出这两种，必须做**只换 prompt、观测全固定**的对照。

关键前提（已核实）：
  * `policy.py:75` 每次 infer 都 `jax.random.split(self._rng)` 推进随机数 →
    **服务端每次推理都重新采样 flow-matching 噪声**，同一观测同一 prompt 两次结果也不同。
    所以必须先用同一 prompt 重复 N 次量出「采样噪声底噪」，否则跨 prompt 的差异没法解读。
  * 参照点最近两点相距 8.4cm、中位 27.1cm；只要跨 prompt 的均值差异远大于底噪，
    prompt 就是有效的。

判据：每个 prompt 重复 N 次 → 算**均值 chunk 末端**。跨 prompt 的均值散布如果
      远大于「均值自身的标准误」(sd/√N)，说明 prompt 真的改变了策略输出。

用法:
  .venv/bin/python probe_prompt_sensitivity.py --reps 15
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.realpath(__file__))
FRAMES = "/home/dgl/lxc/.runlogs/task3_frames"
DUMP = "/home/dgl/lxc/.runlogs/d19goal_chunks.jsonl"

TOOLS = ["电工刀(右)", "螺丝刀", "扳手", "手电筒", "电工刀(左)"]
BOX = "盒子"
CONTROL = ""                      # 空 prompt = 最强的对照


def load_frames(prefix: str) -> dict:
    import cv2
    out = {}
    for env_name, key in (("camera_head_color", "cam_head"),
                          ("camera_wrist_r_color", "cam_wrist_r")):
        p = os.path.join(FRAMES, f"{prefix}_{env_name}.png")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != 480 or rgb.shape[1] != 640:
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
        out[key] = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
    return out


def load_first_state(prompt: str) -> np.ndarray:
    with open(DUMP, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["prompt"] == prompt:
                return np.asarray(r["state"], dtype=np.float32)
    raise KeyError(prompt)


def run_probe(policy, images, state, label, prompts, reps, goals, names):
    """固定观测，逐 prompt 重复采样，返回 {prompt: (reps, 20, 19)}。"""
    print(f"\n{'='*100}\n观测组: {label}   state[7:10]={np.round(state[7:10],4).tolist()}")
    out = {}
    for p in prompts:
        runs = []
        for k in range(reps):
            a = np.asarray(policy.infer({"state": state, "images": images,
                                         "prompt": p})["actions"], dtype=np.float64)
            runs.append(a)
        out[p] = np.stack(runs)                       # (reps, H, D)
        ends = out[p][:, -1, 7:10]                    # 每次的规划末端
        aim = Counter(names[int(np.argmin(np.linalg.norm(goals - e, axis=1)))] for e in ends)
        top = ", ".join(f"{k}:{v}" for k, v in aim.most_common(3))
        print(f"  {p or '<空>':<14} 末端均值={np.round(ends.mean(0),4).tolist()}  "
              f"逐次散布(sd)={ends.std(0).mean()*1000:5.1f}mm   逐次瞄准: {top}")
    return out


def analyse(chunks, prompts, label, goals, names, *, tool_prompts, control):
    """把跨 prompt 的均值差异和同 prompt 的采样噪声放在同一把尺子上比。"""
    means = {p: chunks[p][:, -1, 7:10].mean(0) for p in prompts}
    # 同 prompt 的采样噪声（-1 修正，无偏）
    within_sd = np.mean([chunks[p][:, -1, 7:10].std(0, ddof=1).mean() for p in prompts])
    n = len(chunks[prompts[0]])
    sem = within_sd / np.sqrt(n)

    # 各 prompt 均值相对基线的位移
    ref = means[control]
    print(f"\n--- {label} ---")
    print(f"  同 prompt 采样噪声 sd = {within_sd*1000:.1f}mm → 均值标准误 SEM = {sem*1000:.1f}mm (N={n})")
    print(f"  {'prompt':<26}{'均值末端':>26}{'相对基线位移':>16}{'=几个SEM':>10}")
    deltas = {}
    for p in prompts:
        d = float(np.linalg.norm(means[p] - ref))
        deltas[p] = d
        lab = (p or '<空>')[:24]
        print(f"  {lab:<26}{str(np.round(means[p],4).tolist()):>26}"
              f"{d*1000:>13.1f}mm{d/max(sem,1e-12):>10.1f}")
    all_d = [deltas[p] for p in prompts if p != control]
    print(f"\n  → 换 prompt 引起的均值位移: 中位 {np.median(all_d)*1000:.1f}mm  "
          f"= {np.median(all_d)/max(sem,1e-12):.1f} 个 SEM   最大 {max(all_d)*1000:.1f}mm")
    print(f"  → 参照点最近间距 84mm：位移要接近这个量级，才可能改变'瞄哪件工具'")

    # 逐 prompt 的瞄准分布（这是"它在决定抓哪个"的直接读数，噪声已被 N 次平均压下去）
    print(f"\n  逐 prompt 的瞄准分布 (N={n} 次里各有几次瞄向该点):")
    w = 11
    print(f"    {'prompt':<26}" + "".join(f"{x:>{w}}" for x in names))
    for p in prompts:
        ends = chunks[p][:, -1, 7:10]
        c = Counter(names[int(np.argmin(np.linalg.norm(goals - e, axis=1)))] for e in ends)
        hit = c.get(p, 0) if p in tool_prompts else None
        line = f"    {(p or '<空>')[:24]:<26}" + "".join(f"{c.get(x,0):>{w}}" for x in names)
        if hit is not None:
            line += f"   瞄对 {hit}/{n}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--reps", type=int, default=15, help="每个 prompt 重复次数（量采样噪声用）")
    ap.add_argument("--groups", default="电工刀(右),手电筒",
                    help="用哪几个阶段的真实帧+state 当固定观测（逗号分隔；run_start 表示开局帧）")
    ap.add_argument("--prompts", default="",
                    help="自定义 prompt 列表（用 | 分隔），默认用 5 个工具名 + 空串。"
                         "注意：训练数据的 task 串里**一个工具名都没有**，所以"
                         "「工具名」和「训练用的整句」必须分开测。")
    ap.add_argument("--ref", default="",
                    help="用哪个 prompt 当基线算位移，默认空串；自定义 prompt 时可指最后一个")
    args = ap.parse_args()

    if args.prompts:
        ps = [x.strip() for x in args.prompts.split("|")]
        prompt_list = ps
        # 自由文本模式下没有"正确工具名"可言，把 TOOLS 清空，
        # 于是"瞄对%"整列不打印，只剩下位移与瞄准分布这两个真正可比的东西。
        tool_prompts = []
        control = args.ref if args.ref in ps else ps[-1]
    else:
        prompt_list = TOOLS + [CONTROL]
        tool_prompts = TOOLS
        control = CONTROL
    if control not in prompt_list:
        raise SystemExit(f"--ref「{control}」不在 --prompts 里，无法当基线")

    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    tr = json.load(open(os.path.join(BASE, "tool_refs_d19goal.json"), encoding="utf-8"))
    goals = np.asarray(tr.get("goals") or tr["tool_refs"], dtype=np.float64)
    names = TOOLS + [BOX]
    prompts = prompt_list

    print(f"参照点 {len(goals)} 个   reps={args.reps}   基线 prompt=「{control or '<空>'}」")
    print(f"待测 prompt {len(prompts)} 个:")
    for p in prompts:
        print(f"    「{p}」")

    for g in args.groups.split(","):
        g = g.strip()
        images = load_frames("run_start" if g == "run_start" else f"phase_{g}")
        state = load_first_state(g if g != "run_start" else "电工刀(右)")
        chunks = run_probe(policy, images, state, g, prompts, args.reps, goals, names)
        analyse(chunks, prompts, g, goals, names, tool_prompts=tool_prompts, control=control)

        np.save(f"/tmp/probe_chunks_{g}.npy",
                np.stack([chunks[p] for p in prompts]))   # 存档备查


if __name__ == "__main__":
    main()
