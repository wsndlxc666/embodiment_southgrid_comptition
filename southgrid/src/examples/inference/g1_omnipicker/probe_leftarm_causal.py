#!/usr/bin/env python
"""左臂 dims0~6 换成评测值，预测到底变不变？—— 交错重复测量，把信号和噪声分开。

（probe_state_ood.py 的第 2 段把帧0的右臂配到帧 f 的真值上，构造是错的；
  probe_leftarm_harm.py 的 B/C 列同样混了帧号。这里只做一件事：**除左臂外逐字节相同**。）

对每个帧 f，四个条件（每条件重复 N 次，**交错**执行以抵消漂移）：
   A 训练左臂（= S0[f] 原样）                    基准
   B 评测左臂（局部换成 ik8 实测的 dims0~6）       ← 唯一变量
   C 全黑图（state 同 A）                        视觉有没有用
   D 错相位图（state 同 A，图像取帧 0）            视觉有没有用

两个判据：
  1) 误差 vs 真值（mm）
  2) ‖pred − pred_A‖ / 噪声底 —— 噪声底 = A 内部 各次预测到 A 均值 的平均距离。
     比值 ≲1 表示这次扰动**完全被采样噪声淹没**（模型对它没有响应）。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_leftarm_causal.py [--reps 12]
"""
from __future__ import annotations

import argparse

import numpy as np

from probe_policy_fit import EE, ep_arrays, frames_at
from probe_index_channel import TASKS
from probe_state_ood import EVAL_STATE22

FRAMES = [0, 100, 366, 500, 900, 1166, 1300, 1966]
GRIP = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=args.port)

    S0, A0, ti = ep_arrays(args.ep)
    task = TASKS[ti]
    st_ev = np.array(EVAL_STATE22, dtype=np.float64)
    need = sorted(set(FRAMES) | {0})
    imgs = {c: frames_at(args.ep, c, need) for c in ("cam_head", "cam_wrist_r")}

    print(f"ep{args.ep}  task={task!r}  重复 {args.reps} 次（交错）  16 帧的 chunk")
    print("左臂实测值  训练 " + np.round(S0[0][:7], 4).tolist().__str__())
    print("            评测 " + np.round(st_ev[:7], 4).tolist().__str__())
    print(f"\n{'帧':>6} {'A 训练左臂':>11} {'B 评测左臂':>11} {'Δ左臂/噪声':>11} "
          f"{'C 全黑图':>11} {'Δ黑图/噪声':>11} {'D 错相位图':>11} {'Δ相位/噪声':>11}")

    for f in FRAMES:
        st_a = S0[f].astype(np.float64)
        st_b = st_a.copy(); st_b[:7] = st_ev[:7]
        im_real = {c: imgs[c][f] for c in imgs}
        im_black = {c: np.zeros_like(imgs[c][f]) for c in imgs}
        im_wrong = {c: imgs[c][0] for c in imgs}
        conds = [("A", st_a, im_real), ("B", st_b, im_real),
                 ("C", st_a, im_black), ("D", st_a, im_wrong)]

        preds = {k: [] for k, _, _ in conds}
        for _ in range(args.reps):                      # 交错：外层 reps，内层条件
            for k, st, im in conds:
                a = np.asarray(pol.infer({"state": st.astype(np.float32), "images": im,
                                          "prompt": task})["actions"], dtype=np.float64)
                preds[k].append(a if a.ndim > 1 else a[None])

        gt = A0[f:f + 20]
        n = min(preds["A"][0].shape[0], len(gt))
        ref = np.mean(preds["A"], axis=0)               # A 的均值预测 = 参照点
        noise = np.mean([np.linalg.norm(p[:n, EE] - ref[:n, EE], axis=1).mean() * 1000
                         for p in preds["A"]])
        out = {}
        for k in "ABCD":
            P = np.mean(preds[k], axis=0)
            err = np.sqrt(((P[:n, EE] - gt[:n, EE]) ** 2).sum(1).mean()) * 1000
            dv = np.linalg.norm(P[:n, EE] - ref[:n, EE], axis=1).mean() * 1000
            out[k] = (err, dv)
        print(f"{f:6d} {out['A'][0]:8.1f}mm {out['B'][0]:8.1f}mm {out['B'][1]/noise:10.2f} "
              f"{out['C'][0]:8.1f}mm {out['C'][1]/noise:10.2f} "
              f"{out['D'][0]:8.1f}mm {out['D'][1]/noise:10.2f}")
    print(f"\n  （噪声底 = A 条件内部各次预测到 A 均值的平均距离）")


if __name__ == "__main__":
    main()
