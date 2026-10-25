#!/usr/bin/env python
"""评测时的 state 到底有多"分布外"？以及——左臂那 7 维是不是把模型带偏的真凶？

链条：实测这个 checkpoint **不看图像**（黑图/错相位图的预测与真图无差别，而换 state 的
阳性对照差 500~650mm）。那么模型的行为**完全由 22 维 state 决定**，state 的分布外程度
就直接等于策略失效程度。

已知（ik8 日志）：start_align ik 把右臂摆到 0.00mm/0.00°，dims 7~21 与训练帧 0 逐字节相同，
**只有左臂 dims 0~6 差 2.6mm / ~7°**（左臂在训练里几乎不动，span 只有 1.4~18mm）。
归一化是按分位数做的，"偏离 span" 才是模型真正看到的量。

本脚本两件事：
  1) 把评测帧 0 的 state 逐维换算成"偏离训练跨度的倍数"（用 checkpoint 里的 norm_stats）
  2) **因果检验**：训练 state 与评测 state 交叉替换 dims 0~6，看预测误差怎么变
     A 全训练  B 全评测  C 评测但左臂换回训练  D 训练但左臂换成评测
     若 C ≪ B（把左臂换回来就好了）⇒ 左臂就是真凶
     若 D ≫ A（把左臂换成评测就坏了）⇒ 同上，且方向确认

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_state_ood.py [--reps 8]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from probe_policy_fit import EE, ep_arrays, frames_at
from probe_index_channel import TASKS

NORM = ("/home/dgl/lxc/openpi/checkpoints/pi05_g1_omnipicker_clear5_d22_lora_20k/"
        "clear5_d22_lora_100ep/19000/assets/g1_omnipicker_clear5_d22_100ep/norm_stats.json")
# ik8 实测（d22_ik8.log 的 [start] 行，start_align ik 后 0.00mm/0.00° 的那个 state）
EVAL_STATE22 = [0.1296, 0.9737, 0.303, -0.5209, -0.5349, 0.4655, 0.4751,
                0.1585, -0.2792, 0.4356, -0.6987, 0.5574, 0.3993, 0.2043,
                0.048, 0.048, 0.048, 0.048, 0.4784, -0.0845, 0.51, 1.0]
NAMES = ['L_x', 'L_y', 'L_z', 'L_qw', 'L_qx', 'L_qy', 'L_qz',
         'R_x', 'R_y', 'R_z', 'R_qw', 'R_qx', 'R_qy', 'R_qz',
         'd14', 'd15', 'grip16', 'grip17', 'goal_dist', 'goal_y', 'cyc_dist', 'cyc_idx']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    S0, A0, ti = ep_arrays(args.ep)
    task = TASKS[ti]
    st_tr = S0[0].astype(np.float64)
    st_ev = np.array(EVAL_STATE22, dtype=np.float64)

    ns = json.load(open(NORM))["norm_stats"]["state"]
    q01 = np.array(ns["q01"], dtype=np.float64)
    q99 = np.array(ns["q99"], dtype=np.float64)
    span = q99 - q01
    span[span == 0] = np.nan            # dims 14/15 恒定，除零

    print("=== 1) 评测 state 逐维偏离训练跨度的倍数（1.0 = 整整一个观测跨度）===")
    print(f"{'i':>3} {'名':<9} {'训练q01':>10} {'训练q99':>10} {'跨度':>10} "
          f"{'训练值':>9} {'评测值':>9} {'偏离(倍跨度)':>13}")
    dev = (st_ev - st_tr) / span
    for i in range(22):
        flag = ""
        if not np.isnan(dev[i]) and abs(dev[i]) > 0.5:
            flag = "  ← 超出训练范围" if abs(dev[i]) > 1.0 else "  ← 接近边界"
        print(f"{i:3d} {NAMES[i]:<9} {q01[i]:10.4f} {q99[i]:10.4f} {span[i]:10.5f} "
              f"{st_tr[i]:9.4f} {st_ev[i]:9.4f} {dev[i]:13.2f}{flag}")
    ok = ~np.isnan(dev)
    print(f"\n  |偏离| > 1 倍跨度的维度："
          f"{[NAMES[i] for i in range(22) if ok[i] and abs(dev[i]) > 1.0]}")
    print(f"  左臂 dims0~6 的 |偏离|：{np.round(np.abs(dev[:7]), 2).tolist()}")

    # ---------- 2) 因果：交叉替换左臂 ----------
    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)
    imgs = {c: frames_at(args.ep, c, [0])[0] for c in ("cam_head", "cam_wrist_r")}

    def err_of(st):
        errs = []
        for _ in range(args.reps):
            a = np.asarray(pol.infer({"state": st.astype(np.float32), "images": imgs,
                                      "prompt": task})["actions"], dtype=np.float64)
            a = a if a.ndim > 1 else a[None]
            n = min(len(a), 20)
            errs.append(np.sqrt(((a[:n, EE] - A0[0:n, EE]) ** 2).sum(1).mean()) * 1000)
        return float(np.mean(errs))

    c_ev = st_ev.copy(); c_ev[:7] = st_tr[:7]      # 评测 + 左臂换回训练
    c_tr = st_tr.copy(); c_tr[:7] = st_ev[:7]      # 训练 + 左臂换成评测
    print("\n=== 2) 因果检验：预测 vs 训练帧0 真值（右末端误差，mm）===")
    print("  参考：原地不动 %.1f mm；背平均轨迹 %.1f mm"
          % (np.sqrt(((S0[0, EE] - A0[:20, EE]) ** 2).sum(1).mean()) * 1000, float('nan')))
    for tag, st in (("A 全训练 state", st_tr), ("B 全评测 state", st_ev),
                    ("C 评测state + 左臂换回训练", c_ev),
                    ("D 训练state + 左臂换成评测", c_tr)):
        print(f"  {tag:<28} {err_of(st):7.1f} mm")


if __name__ == "__main__":
    main()
