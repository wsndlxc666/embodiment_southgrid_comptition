#!/usr/bin/env python
"""左臂 7 维的分布外，具体毁掉了什么行为？—— 落到"抓/放"这一层看。

probe_state_ood.py 已证：评测 state 与训练帧0 只差左臂 dims0~6（右臂 teleport 到 0.00mm/0.00°，
dims7~21 逐字节相同），而只换这 7 个数就把右末端预测误差从 62mm 抬到 333mm（换回去 → 67mm）。
本脚本把危害落到行为层：模型还会不会**闭爪**、还会不会**抬到轮间高度**。

选帧原则：要选"真值在该 chunk 内会闭爪"的帧，否则闭爪指标无信息。
真值闭爪帧 = 115 / 515 / 915 / 1315 / 1715（100 集逐字节相同），chunk 长 20 ⇒ 选 100/500/900/1300。

条件：
  A 全训练 state          B 全评测 state（ik8 实测）
  C 评测 state+左臂换回训练   （= 实际该送进去的 state）
  D 训练 state+左臂换成评测

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_leftarm_harm.py [--reps 8]
"""
from __future__ import annotations

import argparse

import numpy as np

from probe_policy_fit import EE, ep_arrays, frames_at
from probe_index_channel import TASKS
from probe_state_ood import EVAL_STATE22

FRAMES = [100, 500, 900, 1300]        # 真值闭爪分别在这些帧的 +15 附近
GRIP = 16


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
    st_ev = np.array(EVAL_STATE22, dtype=np.float64)
    imgs = {c: frames_at(args.ep, c, FRAMES) for c in ("cam_head", "cam_wrist_r")}

    def first_close(g):
        w = np.where(np.asarray(g) > 0.5)[0]
        return int(w[0]) if len(w) else -1

    print(f"ep{args.ep}  task={task!r}  重复 {args.reps} 次  chunk 长 20")
    print("A 全训练  B 全评测  C 评测+左臂换回训练  D 训练+左臂换评测\n")
    hdr = (f"{'帧':>6} {'条件':<22} {'右末端误差':>10} {'闭爪帧':>7} {'20步净位移':>11} "
           f"{'z 跨度':>8}   真值")
    print(hdr)

    for f in FRAMES:
        st_tr = S0[f].astype(np.float64)
        gt = A0[f:f + 20]
        im = {c: imgs[c][f] for c in imgs}      # 只取该帧（frames_at 返回的是按帧号索引的 dict）
        gtc, gtz = first_close(gt[:, GRIP]), (gt[:, 9].max() - gt[:, 9].min()) * 1000
        conds = [("A 全训练", st_tr),
                 ("B 全评测", st_ev),
                 ("C 评测+左臂换回训练", np.concatenate([st_tr[:7], st_ev[7:]])),
                 ("D 训练+左臂换评测", np.concatenate([st_ev[:7], st_tr[7:]]))]
        for tag, st in conds:
            errs, cls, net, zsp = [], [], [], []
            for _ in range(args.reps):
                a = np.asarray(pol.infer({"state": st.astype(np.float32), "images": im,
                                          "prompt": task})["actions"], dtype=np.float64)
                a = a if a.ndim > 1 else a[None]
                n = min(len(a), 20)
                errs.append(np.sqrt(((a[:n, EE] - gt[:n, EE]) ** 2).sum(1).mean()) * 1000)
                cls.append(first_close(a[:n, GRIP]))
                net.append(np.linalg.norm(a[n - 1, EE] - a[0, EE]) * 1000)
                zsp.append((a[:n, 9].max() - a[:n, 9].min()) * 1000)
            # 闭爪帧用众数（离散）
            vals, cnt = np.unique(cls, return_counts=True)
            cmode = int(vals[cnt.argmax()])
            print(f"{f:6d} {tag:<22} {np.mean(errs):7.1f}mm {cmode:7d} "
                  f"{np.mean(net):8.1f}mm {np.mean(zsp):6.1f}mm   "
                  f"闭爪@{gtc}  净位移{np.linalg.norm(gt[-1,EE]-gt[0,EE])*1000:.1f}mm "
                  f"z跨度{gtz:.1f}mm")
        print()


if __name__ == "__main__":
    main()
