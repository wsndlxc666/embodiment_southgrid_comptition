#!/usr/bin/env python
"""这个策略到底"学会了"没有？—— 用数据集自己的 action 当真值，跟三条基线比。

为什么必须比基线：100 集是**同一条轨迹 ±40mm**（probe_dataset_diversity.py 实测：
每帧跨集 std 10/15/6mm，两两 RMS 20~53mm，而轨迹自身幅度 530~685mm）。在这种数据上，
一个**完全不看观测、只照背平均轨迹**的策略，误差就已经只有 ~40mm。所以"误差 55mm"
这种数看着不大，其实可能等于什么都没学。判据必须是：

    误差(模型)  vs  误差(背平均轨迹)  vs  误差(原地不动)

  · 若 模型误差 ≫ 背平均  → 模型连均值都没学会（欠拟合/优化问题）
  · 若 模型误差 ≈ 背平均  → 模型就是个**开环轨迹复读机**，观测里那些信息它没用
  · 若 模型误差 ≪ 背平均  → 它真的在用观测，问题在别处（环境不匹配等）

同时报"重复之间的标准差"，把采样噪声和真实误差分开（噪声 ~86mm，必须重复平均）。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_policy_fit.py [--reps 8]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
EVAL_FRAMES = "/home/dgl/lxc/.runlogs/task3_frames_ik8"
CAM_KEY = {"cam_head": "observation.images.cam_head",
           "cam_wrist_r": "observation.images.cam_wrist_r"}
TASKS = {0: "把工具全部收进箱子里",
         1: "把桌面上的工具都收进工具箱",
         2: "把所有工具从左到右放回工具箱"}
EE = slice(7, 10)
PROBE_FRAMES = [0, 200, 366, 700, 1166, 1600, 1966]


def ep_arrays(ep):
    df = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet",
                         columns=["observation.state", "action", "task_index"])
    S = np.array([list(s) for s in df["observation.state"]], dtype=np.float64)
    A = np.array([list(a) for a in df["action"]], dtype=np.float64)
    return S, A, int(df["task_index"].iloc[0])


def frames_at(ep, cam, wanted):
    """一次解码，取出 wanted 里的所有帧（HEVC 顺序解码，别每帧重开）。"""
    import av
    p = f"{DS}/videos/chunk-000/{CAM_KEY[cam]}/episode_{ep:06d}.mp4"
    want = set(wanted)
    out = {}
    with av.open(p) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                out[i] = np.transpose(np.ascontiguousarray(fr.to_ndarray(format="rgb24")),
                                      (2, 0, 1))
                if len(out) == len(want):
                    break
    return out


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
    eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                 for p in glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
    As = np.stack([ep_arrays(e)[1] for e in eps])       # (100, 2101, 22)
    Amean = As.mean(axis=0)
    print(f"真值来自 ep{args.ep}（task={task!r}）；均值轨迹由全部 {len(eps)} 集平均得到；"
          f"重复 {args.reps} 次")

    imgs = {c: frames_at(args.ep, c, PROBE_FRAMES) for c in CAM_KEY}
    missing = [f for f in PROBE_FRAMES if f not in imgs["cam_head"]]
    if missing:
        print(f"⚠ 取不到的帧 {missing}")

    print(f"\n{'帧':>6} {'模型(chunk均值)':>16} {'背平均轨迹':>12} {'原地不动':>10} "
          f"{'跨集tube':>10} {'模型/背平均':>12} {'重复std':>9} {'夹爪闭爪帧(模型/真值)':>22}")
    rows = []
    for f in PROBE_FRAMES:
        if f not in imgs["cam_head"]:
            continue
        obs = {"state": S0[f].astype(np.float32),
               "images": {k: imgs[k][f] for k in CAM_KEY}, "prompt": task}
        outs = []
        for _ in range(args.reps):
            a = np.asarray(pol.infer(obs)["actions"], dtype=np.float64)
            outs.append(a if a.ndim > 1 else a[None])   # 保留整块 (T, 22)
        O = np.stack(outs)                      # (reps, T, 22)
        n = O.shape[1]
        avg = O.mean(axis=0)                    # 平均掉采样噪声
        gt = A0[f:f + n]                        # 真值 action
        m = min(n, len(gt))
        e_model = np.sqrt(((avg[:m, EE] - gt[:m, EE]) ** 2).sum(1).mean()) * 1000
        e_mean = np.sqrt(((Amean[f:f + m, EE] - gt[:m, EE]) ** 2).sum(1).mean()) * 1000
        e_stay = np.sqrt(((S0[f, EE] - gt[:m, EE]) ** 2).sum(1).mean()) * 1000
        e_tube = np.sqrt(((As[:, f:f + m, EE] - gt[:m, EE]) ** 2).sum(-1).mean()) * 1000
        rep_std = O[:, :, EE].std(axis=0).mean() * 1000
        # 夹爪：模型平均预测里第一次闭爪的 k，vs 真值的
        def first_close(g):
            w = np.where(g > 0.5)[0]
            return int(w[0]) if len(w) else -1
        k_m, k_g = first_close(avg[:m, 16]), first_close(gt[:m, 16])
        rows.append((f, e_model, e_mean, e_stay, e_tube))
        print(f"{f:6d} {e_model:13.1f}mm {e_mean:9.1f}mm {e_stay:7.1f}mm {e_tube:7.1f}mm "
              f"{e_model/e_mean:12.2f} {rep_std:6.1f}mm {str(k_m)+' / '+str(k_g):>22}")

    if rows:
        r = np.array(rows)
        print(f"\n  平均：模型 {r[:,1].mean():.1f}mm | 背平均轨迹 {r[:,2].mean():.1f}mm | "
              f"原地不动 {r[:,3].mean():.1f}mm | 跨集tube半径 {r[:,4].mean():.1f}mm")
        print(f"  ⇒ 模型误差 / 背平均误差 = {r[:,1].mean()/r[:,2].mean():.2f}"
              f"（≥1 说明模型没比'照背平均轨迹'更好；≪1 才说明它真在用观测）")


if __name__ == "__main__":
    main()
