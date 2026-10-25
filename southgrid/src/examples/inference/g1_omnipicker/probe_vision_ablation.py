#!/usr/bin/env python
"""视觉分支到底在不在干活？—— 把图像换成黑的/换成别的时刻的，看误差变不变。

动机：轮中复读误差已经只有 3~10mm，比"背平均轨迹"好 4~7 倍，说明模型确实在用**某种**
输入。但那个输入可能是本体感知（手臂位姿本身就能大致确定处在第几阶段），而不是图像。
如果视觉分支其实是死的，那"训不出来"就有了一个训练侧的硬原因，而不是数据问题。

三个条件（其余逐字节相同、每条件重复 N 次取均值）：
   A 真图            —— 基准
   B 全黑图          —— 有像素依赖就会变差
   C 错相位真图      —— 用**同一集别的时刻**的图（像素仍在分布内，只是阶段不对）
                        模型若真的从画面读阶段，C 应该显著差于 A；若忽略画面，C≈A
对照：把 state 也一起换掉（错帧的 state+图）应当**必然**变差 —— 这是"测试本身有效"的阳性对照。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_vision_ablation.py [--reps 8]
"""
from __future__ import annotations

import argparse

import numpy as np

from probe_policy_fit import EE, ep_arrays, frames_at
from probe_index_channel import TASKS

FRAMES = [366, 700, 1600]          # 366=轮次切换处（模型最差），700/1600=轮中（模型最好）
MISMATCH = {366: 0, 700: 0, 1600: 0}   # 错相位用帧 0 的图（阶段完全不同）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)

    S0, A0, ti = ep_arrays(args.ep)
    task = TASKS[ti]
    need = sorted(set(FRAMES) | set(MISMATCH.values()))
    imgs = {c: frames_at(args.ep, c, need) for c in ("cam_head", "cam_wrist_r")}

    def infer(st, im):
        a = np.asarray(pol.infer({"state": st.astype(np.float32), "images": im,
                                  "prompt": task})["actions"], dtype=np.float64)
        return a if a.ndim > 1 else a[None]

    print(f"ep{args.ep}  task={task!r}  重复 {args.reps} 次")
    print(f"\n{'帧':>5} {'A 真图':>12} {'B 全黑图':>12} {'C 错相位图':>13} "
          f"{'阳性对照(错state+错图)':>22}   判读")
    for f in FRAMES:
        real = {c: imgs[c][f] for c in imgs}
        black = {c: np.zeros_like(imgs[c][f]) for c in imgs}
        m = MISMATCH[f]
        wrong = {c: imgs[c][m] for c in imgs}

        res = {}
        for tag, st, im in (("A", S0[f], real), ("B", S0[f], black),
                            ("C", S0[f], wrong), ("D", S0[m], wrong)):
            errs = []
            for _ in range(args.reps):
                ch = infer(st, im)
                n = min(len(ch), 20)
                errs.append(np.sqrt(((ch[:n, EE] - A0[f:f + n, EE]) ** 2).sum(1).mean()) * 1000)
            res[tag] = float(np.mean(errs))
        verdict = []
        if res["B"] > res["A"] * 1.5:
            verdict.append("有像素依赖")
        else:
            verdict.append("**黑图也不影响→视觉可能没用**")
        if res["C"] > res["A"] * 1.5:
            verdict.append("读了阶段")
        else:
            verdict.append("**错相位也不影响→没从画面读阶段**")
        print(f"{f:5d} {res['A']:9.1f}mm {res['B']:9.1f}mm {res['C']:10.1f}mm "
              f"{res['D']:19.1f}mm   {'；'.join(verdict)}")


if __name__ == "__main__":
    main()
