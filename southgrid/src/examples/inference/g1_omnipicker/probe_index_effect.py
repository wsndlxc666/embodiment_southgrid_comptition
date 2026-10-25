#!/usr/bin/env python
"""重复测量：只改 state 的 dim19/20/21，模型输出到底动不动？（判据要压过采样噪声）

上一版 probe_index_channel.py 每种条件只跑 1 次，得到的变化 18~73mm，而**同一输入重复 8 次
的极差就有 86mm(y) / 120mm(模长)** —— 单样本读不出来。所以这里做重复测量：
每种 idx 条件各跑 N 次，**交错**执行（外层循环是重复、内层是条件，抵消 GPU 状态慢漂移），
报 均值±标准误，并算 y 对 goal_y(dim19) 的回归斜率。

判据：
  · 若各条件的均值**分得开**（差 ≫ 标准误）且 **y 随 goal_y 单调** → 模型确实读这条通道
  · 若各条件均值几乎重合、差异被标准误吞掉 → 模型没在读它

只改 dim18..21（goal_dist/goal_y/cycle_dist/cycle_idx），图像与臂位姿逐字节冻结。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_index_effect.py [--reps 12]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from probe_index_channel import (CYCLE_START, DS, EVAL_FRAMES, TASKS, TOOL_REF,
                                 ep_data, png_frame, video_frame)


def chw(img):
    return np.transpose(np.ascontiguousarray(img), (2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)

    S, A, ti = ep_data(0)
    pr = TASKS[ti]
    # 条件：dim18..21 取自 ep0 每轮起始帧的真值
    conds = [(idx, S[f, 18:22].copy()) for f, idx in CYCLE_START]

    sets = {"分布内(训练帧0)": {
        "cam_head": chw(video_frame(0, "cam_head", 0)),
        "cam_wrist_r": chw(video_frame(0, "cam_wrist_r", 0))}}
    try:
        sets["分布外(评测ik8)"] = {
            "cam_head": chw(png_frame(os.path.join(EVAL_FRAMES,
                                                   "attempt_start_camera_head_color.png"))),
            "cam_wrist_r": chw(png_frame(os.path.join(EVAL_FRAMES,
                                                      "attempt_start_camera_wrist_r_color.png")))}
    except Exception as exc:
        print(f"（跳过分布外那组：{exc}）")

    print(f"task 串 = {pr!r}   重复次数 = {args.reps}（交错执行）")
    print(f"各条件的 goal_y(dim19): " + "  ".join(f"idx{idx}:{c[1]:+.4f}" for idx, c in conds))

    for tag, imgs in sets.items():
        print(f"\n################ {tag} ################")
        acc: dict[int, list] = {idx: [] for idx, _ in conds}
        for _ in range(args.reps):
            for idx, tail in conds:
                st = S[0].copy()
                st[18:22] = tail
                a = np.asarray(pol.infer({"state": st, "images": imgs, "prompt": pr})["actions"],
                               dtype=np.float32)
                if a.ndim == 1:
                    a = a[None]
                acc[idx].append(a)
        print(f"  {'idx':>3} {'goal_y':>9} | {'chunk[0] 右末端 (均值±标准误, mm)':<34} "
              f"| {'chunk均值 7:10 y':>18} | {'chunk[-1] y':>18}")
        ys0, ysm, ysN, gy = [], [], [], []
        for idx, tail in conds:
            arr = np.stack(acc[idx])                    # (reps, T, 22)
            a0 = arr[:, 0, 7:10]                        # 首动作
            am = arr[:, :, 7:10].mean(axis=1)           # 整块均值
            aN = arr[:, -1, 7:10]                       # 块末
            sem = a0.std(0) / np.sqrt(len(arr))
            print(f"  {idx:3d} {tail[1]:+9.4f} | "
                  f"x{a0[:,0].mean():+.4f}±{sem[0]*1000:4.1f} "
                  f"y{a0[:,1].mean():+.4f}±{sem[1]*1000:4.1f} "
                  f"z{a0[:,2].mean():+.4f}±{sem[2]*1000:4.1f} | "
                  f"{am[:,1].mean():+.4f}±{am[:,1].std()/np.sqrt(len(arr))*1000:5.1f} | "
                  f"{aN[:,1].mean():+.4f}±{aN[:,1].std()/np.sqrt(len(arr))*1000:5.1f}")
            ys0.append(a0[:, 1].mean()); ysm.append(am[:, 1].mean())
            ysN.append(aN[:, 1].mean()); gy.append(tail[1])
        gy = np.array(gy)
        for nm, ys in (("chunk[0]", ys0), ("chunk均值", ysm), ("chunk[-1]", ysN)):
            ys = np.array(ys)
            sl = np.polyfit(gy, ys, 1)[0]               # Δy / Δgoal_y
            r = np.corrcoef(gy, ys)[0, 1]
            print(f"  → {nm:9s} y 对 goal_y 的斜率 {sl:+.3f}（1.0 = 完全跟随），"
                  f"相关 r={r:+.3f}，首末条件差 {(ys[-1]-ys[0])*1000:+7.1f} mm")
        # 组内噪声参照：同条件的标准误平均
        se = np.mean([np.stack(acc[i])[:, 0, 8].std() / np.sqrt(args.reps) for i in acc])
        print(f"  → 单条件标准误均值 {se*1000:.1f} mm（条件间差异要与它比）")


if __name__ == "__main__":
    main()
