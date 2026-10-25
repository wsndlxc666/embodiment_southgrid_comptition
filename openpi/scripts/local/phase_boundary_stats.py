#!/usr/bin/env python
"""第 0 步（只读）：把 d22 数据集里的「相位」量化出来。

要回答的三件事：
  1. 100 集的 cycle_idx(dim 21) 序列是否逐字节一致 ⇒ 能否用统一帧号切分
  2. 相位边界（idx 变化点）具体是哪几帧，各轮各占多少帧
  3. 边界 ±20 帧的样本占全集多大比例 —— 这个比例就是"相位通道能拿到多少梯度"的上界
     （训练时每个样本是「某一帧 + 后续 20 步」，所以 20 帧 = 一个 chunk 的长度）

只读 parquet，不碰视频，不碰仿真。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
DIM = 22
IDX_DIM = 21
CHUNK = 20          # action_horizon


def load_ep(path: str) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(path, columns=["observation.state", "action"])
    st = np.stack([np.asarray(v, dtype=np.float64) for v in t["observation.state"].to_pylist()])
    ac = np.stack([np.asarray(v, dtype=np.float64) for v in t["action"].to_pylist()])
    return st, ac


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-*", "episode_*.parquet")))
    print(f"数据集 {DATA}\n集数 {len(files)}")

    states, actions = [], []
    for f in files:
        st, ac = load_ep(f)
        states.append(st)
        actions.append(ac)

    S = np.stack(states)      # (E, T, D)
    A = np.stack(actions)
    E, T, D = S.shape
    print(f"state {S.shape}   action {A.shape}")

    # ---------- 1) dim 21 是否逐集一致 ----------
    idx0 = S[0, :, IDX_DIM]
    same = all(np.array_equal(S[e, :, IDX_DIM], idx0) for e in range(E))
    print(f"\n[1] cycle_idx 逐集一致: {same}")
    if not same:
        n_diff = sum(1 for e in range(E) if not np.array_equal(S[e, :, IDX_DIM], idx0))
        print(f"    !! 有 {n_diff}/{E} 集不同，统一帧号切分不成立，下面的统计按 ep0 给")

    # action 端 dim21 应等于下一帧的 state dim21（数据集是"未来 20 帧"对齐）
    ok_next = np.allclose(A[:, :-1, IDX_DIM], S[:, 1:, IDX_DIM])
    print(f"[1] action[:,21] == state[t+1,21]: {ok_next}")

    # ---------- 2) 边界与各轮帧数 ----------
    chg = np.where(np.diff(idx0) != 0)[0] + 1      # 变化点（新相位的第 0 帧）
    print(f"\n[2] 相位边界（新相位首帧）: {chg.tolist()}")
    print(f"    idx 取值序列: {idx0[0]:.0f} -> {idx0[-1]:.0f}，共 {len(chg)} 次切换")

    seg_starts = np.concatenate([[0], chg])
    seg_ends = np.concatenate([chg, [T]])
    print(f"    {'相位':>4} {'起':>6} {'止':>6} {'帧数':>6}   {'首帧 idx':>8}")
    for k, (a, b) in enumerate(zip(seg_starts, seg_ends)):
        print(f"    {k:>4} {a:>6} {b - 1:>6} {b - a:>6}   {idx0[a]:>8.0f}")

    # ---------- 3) 边界 ±CHUNK 帧的样本占比 ----------
    mask = np.zeros(T, dtype=bool)
    for c in chg:
        mask[max(0, c - CHUNK):min(T, c + CHUNK + 1)] = True
    n_b = int(mask.sum())
    print(f"\n[3] 边界 ±{CHUNK} 帧: {n_b}/{T} = {100.0 * n_b / T:.2f}%")
    print(f"    若按「变化点之后 {CHUNK} 帧」只取窗口: "
          f"{len(chg) * CHUNK}/{T} = {100.0 * len(chg) * CHUNK / T:.2f}%")

    # 逐维分布：这 4.8% 里，几何上哪里最需要相位
    print(f"\n[3] dim 19/20/21 在边界窗口内 vs 外的取值跨度（物理量，未归一化）")
    for d, nm in ((19, "goal_y"), (20, "cycle_dist"), (21, "cycle_idx")):
        inw = S[0][mask, d]
        outw = S[0][~mask, d]
        print(f"    dim{d:>2} {nm:<11} 窗口内 [{inw.min():9.4f}, {inw.max():9.4f}] "
              f"std={inw.std():8.4f} | 窗口外 [{outw.min():9.4f}, {outw.max():9.4f}] "
              f"std={outw.std():8.4f}")

    # 关键：goal_y 在窗口内是否真的会变（会变才说明"必须读 idx"）
    gy = S[0][:, 19]
    uniq_in = np.unique(np.round(gy[mask], 4))
    print(f"\n[3] 窗口内 goal_y 出现过的不同取值: {uniq_in.tolist()}")


if __name__ == "__main__":
    main()
