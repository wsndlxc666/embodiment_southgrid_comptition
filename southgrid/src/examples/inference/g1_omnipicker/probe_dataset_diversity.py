#!/usr/bin/env python
"""这 100 集到底提供了多少"不同的示范"？—— 决定"为什么训不出策略"的根问题。

疑惑：数据"同类型且有细微差异"，逻辑上学会五件工具不该难。所以要先把"细微差异"量出来：
模仿学习学到的是 p(action | 观测)。如果 100 集在**观测空间**几乎重合、在**动作空间**几乎
重合，那有效样本量不是 100 而是 ≈1，多出来的 99 集只是把同一条梯度方向重复了 99 次 ——
既没有覆盖标称轨迹外的状态，也没有恢复行为可学。

本脚本测四件事（都在 state/action 上，不碰视频）：
  1) 夹爪跳变帧、cycle_idx 变化点：100 集是否逐帧相同
  2) 每帧上"跨集的离散度"：右末端位置在各集之间的标准差，按时间平均
  3) 轨迹空间的两两距离：把每集右末端轨迹当一条曲线，算集间 RMS 距离
  4) 起点与终点的跨集跨度（对比亚毫米噪声 vs 真实变化）

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_dataset_diversity.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
EE = slice(7, 10)          # 右末端位置（base 系）
GRIP = 16                  # 右爪归一化值


def load(ep):
    df = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet",
                         columns=["observation.state", "action"])
    return (np.array([list(s) for s in df["observation.state"]], dtype=np.float64),
            np.array([list(a) for a in df["action"]], dtype=np.float64))


def main():
    eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                 for p in glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
    print(f"共 {len(eps)} 集：{eps[0]}..{eps[-1]}")

    S = {}
    for e in eps:
        S[e] = load(e)[0]
    L = min(len(v) for v in S.values())
    print(f"各集长度 {min(len(v) for v in S.values())}..{max(len(v) for v in S.values())}"
          f"（取前 {L} 帧对齐比较）")

    # ---------- 1) 事件帧是否逐集相同 ----------
    print("\n########## 1) 关键事件帧在 100 集之间是否相同 ##########")
    def events(sig, thr=None):
        if thr is None:
            return tuple(np.where(np.abs(np.diff(sig)) > 0.5)[0] + 1)
        return tuple(np.where(np.diff((sig > thr).astype(int)) != 0)[0] + 1)

    grip_ev = {e: events(S[e][:L, GRIP]) for e in eps}
    idx_ev = {e: tuple(np.where(np.diff(S[e][:L, 21]) > 0)[0] + 1) for e in eps}
    ng = {len(v) for v in grip_ev.values()}
    ni = {len(v) for v in idx_ev.values()}
    uniq_g = {v for v in grip_ev.values()}
    uniq_i = {v for v in idx_ev.values()}
    print(f"  夹爪跳变帧：{'**100 集完全相同**' if len(uniq_g)==1 else f'{len(uniq_g)} 种不同的序列'}"
          f"   序列长度集合 {ng}")
    print(f"    → {sorted(uniq_g)[0] if len(uniq_g)==1 else sorted(uniq_g)[:2]}")
    print(f"  cycle_idx 变化点：{'**100 集完全相同**' if len(uniq_i)==1 else f'{len(uniq_i)} 种'}"
          f"   序列长度集合 {ni}")
    print(f"    → {sorted(uniq_i)[0] if len(uniq_i)==1 else sorted(uniq_i)[:2]}")

    # ---------- 2) 逐帧跨集离散度 ----------
    print("\n########## 2) 每一帧上「跨集」的离散度（右末端，mm）##########")
    P = np.stack([S[e][:L, EE] for e in eps])        # (n_ep, L, 3)
    sd = P.std(axis=0)                                # (L, 3) 各帧跨集标准差
    rng = P.max(axis=0) - P.min(axis=0)
    print(f"  逐帧跨集标准差：x {sd[:,0].mean()*1000:.2f}  y {sd[:,1].mean()*1000:.2f}  "
          f"z {sd[:,2].mean()*1000:.2f}  mm（按时间平均）")
    print(f"  逐帧跨集极差  ：x {rng[:,0].mean()*1000:.2f}  y {rng[:,1].mean()*1000:.2f}  "
          f"z {rng[:,2].mean()*1000:.2f}  mm")
    # 轨迹本身的运动幅度作对照
    self_rng = P.mean(axis=0).max(axis=0) - P.mean(axis=0).min(axis=0)
    print(f"  对照：一条轨迹自身的运动幅度 x {self_rng[0]*1000:.1f}  y {self_rng[1]*1000:.1f}  "
          f"z {self_rng[2]*1000:.1f}  mm")
    print(f"  ⇒ 信噪比（运动幅度 / 跨集噪声）："
          f"x {self_rng[0]/sd[:,0].mean():.0f}×  y {self_rng[1]/sd[:,1].mean():.0f}×  "
          f"z {self_rng[2]/sd[:,2].mean():.0f}×")

    # ---------- 3) 集间轨迹 RMS 距离 ----------
    print("\n########## 3) 任意两集的轨迹差多少（RMS，mm）##########")
    sub = eps[::max(1, len(eps)//20)][:20]
    Q = np.stack([S[e][:L, EE] for e in sub])
    d = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            d.append(np.sqrt(((Q[i] - Q[j]) ** 2).sum(axis=1).mean()) * 1000)
    d = np.array(d)
    print(f"  抽 {len(sub)} 集，{len(d)} 对：RMS 距离 min {d.min():.2f}  median "
          f"{np.median(d):.2f}  max {d.max():.2f}  mm")
    print(f"  ⇒ 100 集里最不像的两集，整条轨迹平均也只差 {d.max():.2f} mm")

    # ---------- 4) 起点/终点跨度 ----------
    print("\n########## 4) 起点与终点的跨集跨度（mm）##########")
    for nm, f in (("起点 帧0", 0), ("终点 末帧", L - 1)):
        p = np.stack([S[e][f, EE] for e in eps])
        print(f"  {nm}: 均值 [{p[:,0].mean():+.4f} {p[:,1].mean():+.4f} {p[:,2].mean():+.4f}]"
              f"  各轴跨度 {np.round((p.max(0)-p.min(0))*1000,2).tolist()} mm  "
              f"std {np.round(p.std(0)*1000,2).tolist()} mm")

    # ---------- 5) 动作维度上的多样性 ----------
    print("\n########## 5) 动作空间：100 集的动作是否也几乎重合 ##########")
    A = np.stack([load(e)[1][:L, EE] for e in eps[::5]][:20])
    sda = A.std(axis=0)
    print(f"  逐帧跨集动作标准差（抽 20 集）：x {sda[:,0].mean()*1000:.2f}  "
          f"y {sda[:,1].mean()*1000:.2f}  z {sda[:,2].mean()*1000:.2f}  mm")
    print(f"  ⇒ 每帧该给什么动作，跨集的答案也只有毫米级分歧")


if __name__ == "__main__":
    main()
