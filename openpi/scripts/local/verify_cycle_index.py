#!/usr/bin/env python
"""校验 cycle_index 三个组件：布局 / 过采样采样器 / 反事实增强。

必须写成 .py 文件跑 —— 从 stdin/heredoc 驱动会让 DataLoader 的 spawn worker 找不到
`<stdin>` 而崩（memory: openpi-dataloader-spawn-stdin）。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_cycle_index.py
"""
from __future__ import annotations

import numpy as np

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.training import cycle_index as CI

REPO = "g1_omnipicker_clear5_d22_100ep"
WINDOW = 60
WEIGHT = 10.0
PROB = 0.5
OK = True


def check(cond: bool, msg: str) -> None:
    global OK
    OK = OK and bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")


def main() -> None:
    meta = lerobot_dataset.LeRobotDatasetMetadata(REPO)
    ds = lerobot_dataset.LeRobotDataset(REPO, delta_timestamps={"action": [t / meta.fps for t in range(20)]})
    print(f"数据集 {REPO}: {len(ds)} 帧, {meta.total_episodes} 集, fps={meta.fps}")

    # ---------------- 1) 布局 ----------------
    print("\n### 1) build_cycle_layout（window=60）")
    layout = CI.build_cycle_layout(ds, window=WINDOW, seed=0)
    n = len(layout)
    cps = layout.change_points
    same = all(c == cps[0] for c in cps)
    print(f"  全局帧数 {n}，集数 {layout.n_episodes}")
    print(f"  每集变化点（ep0）: {cps[0]}")
    check(n == meta.total_episodes * 2101, f"全局帧数 {n} == 100 x 2101")
    check(same, f"100 集的变化点逐集一致")
    check(cps[0] == [366, 766, 1166, 1566, 1966], f"变化点 == 366/766/1166/1566/1966，实际 {cps[0]}")
    exp_b = 5 * WINDOW * layout.n_episodes
    check(layout.n_boundary == exp_b, f"边界样本 {layout.n_boundary} == 5 变化点 x {WINDOW} 帧 x 100 集 = {exp_b}")
    check(abs(layout.n_boundary / n - 0.1428) < 1e-3, f"边界占比 {100*layout.n_boundary/n:.2f}%")

    # 边界帧必须**紧跟在变化点之后**，且不能越过相位末尾
    e0 = 0
    fr = layout.frame[layout.ep == e0]
    b0 = layout.boundary[layout.ep == e0]
    got = np.flatnonzero(b0)
    want = np.concatenate([np.arange(c, c + WINDOW) for c in cps[0]])
    check(np.array_equal(got, want), "ep0 的边界帧号 == 5 个 [c, c+60) 区间")

    # ---------------- 2) 反事实供体 ----------------
    print("\n### 2) 供体值（dims 19/20）必须来自**别的相位**的真实帧")
    st = CI._load_states(CI.find_lerobot_dataset(ds))
    dep, dfr = layout.ep == e0, layout.frame
    b_idx = np.flatnonzero(layout.boundary & (layout.ep == e0))
    dv = layout.donor_vals[b_idx]
    # 供体的 goal_y 应当等于它所在相位的 goal_y（同一相位内 goal_y 恒定）
    phase_gy = {c: st[e0 * 2101 + c, CI.GOAL_Y_DIM] for c in [0, 366, 766, 1166, 1566, 1966]}
    gy_of = lambda f: phase_gy[max(k for k in phase_gy if k <= f)]
    self_gy = np.array([gy_of(f) for f in dfr[b_idx]])
    d_gy = np.abs(dv[:, 0] - self_gy)
    diff_frac = float((d_gy > 1e-9).mean())
    print(f"  供体 goal_y 与自身相位 goal_y 不同的比例 = {100*diff_frac:.1f}%")
    check(diff_frac > 0.6, "多数供体确实来自别的相位（>60%）")
    print(f"  供体 goal_y 取值范围 {np.round(np.unique(dv[:,0]),4).tolist()}")
    print(f"  自身相位 goal_y 取值 {np.round(np.unique(self_gy),4).tolist()}")

    # ---------------- 3) 采样器 ----------------
    print(f"\n### 3) CycleBoundarySampler（weight={WEIGHT}）")
    s = CI.CycleBoundarySampler(layout, weight=WEIGHT, seed=0)
    check(len(s) == n + int(WEIGHT - 1) * layout.n_boundary,
          f"epoch 长度 {len(s)} == {n} + 9 x {layout.n_boundary}")
    a = np.array(list(iter(s)), dtype=np.int64)
    check(a.shape[0] == len(s), "迭代产出的数量与 __len__ 一致")
    check(len(np.unique(a)) == n, "所有原始样本都出现过")
    counts = np.bincount(a, minlength=n)
    check((counts[layout.boundary] == int(WEIGHT)).all(), f"每个边界帧恰好出现 {int(WEIGHT)} 次")
    check((counts[~layout.boundary] == 1).all(), "每个非边界帧恰好出现 1 次")
    # 两轮之间顺序要不一样
    s2 = CI.CycleBoundarySampler(layout, weight=WEIGHT, seed=0)
    b1 = np.array(list(iter(s2))[:200])
    b2 = np.array(list(iter(s2))[:200])
    check(not np.array_equal(b1, b2), "每次 __iter__ 重新打乱（两轮前缀不同）")
    check((np.bincount(b1, minlength=n)[layout.boundary].sum() > 0), "首 200 个里含边界样本")
    print(f"  {s.describe()}")

    # ---------------- 4) 反事实增强 ----------------
    print(f"\n### 4) CycleCounterfactualDataset（prob={PROB}）")
    cf = CI.CycleCounterfactualDataset(ds, layout, prob=PROB, seed=0)
    hit = np.array([cf._hit(i) for i in range(len(cf))])
    check(not hit[~layout.boundary].any(), "非边界帧一定不被改")
    br = hit[layout.boundary].mean()
    print(f"  边界帧命中率 {100*br:.1f}%（期望 ≈ {100*PROB:.0f}%）")
    check(abs(br - PROB) < 0.02, "命中率接近 prob")

    # 真跑几条：命中的必须只改 19/20，其余（含 21）逐字节不变
    tested = 0
    for i in np.flatnonzero(layout.boundary)[:400]:
        raw = ds[int(i)]["observation.state"]
        got = cf[int(i)]["observation.state"]
        raw_a, got_a = np.asarray(raw), np.asarray(got)
        if cf._hit(int(i)):
            tested += 1
            same_mask = np.ones(raw_a.shape[0], dtype=bool)
            same_mask[list(CI.CF_DIMS)] = False
            if not np.allclose(raw_a[same_mask], got_a[same_mask], atol=0):
                check(False, f"帧 {i}: 命中了却动了 dims19/20 之外的维")
                break
            if not np.allclose(got_a[list(CI.CF_DIMS)], layout.donor_vals[int(i)], atol=0):
                check(False, f"帧 {i}: 换进去的不是供体值")
                break
            if raw_a[CI.CYCLE_IDX_DIM] != got_a[CI.CYCLE_IDX_DIM]:
                check(False, f"帧 {i}: cycle_idx 被改了")
                break
    else:
        check(True, f"抽查 {tested} 个命中帧：只改 dims19/20、cycle_idx 原样、其余逐字节相同")
    print(f"  例：帧 {int(np.flatnonzero(hit & layout.boundary)[0])} "
          f"原 goal_y={np.asarray(ds[int(np.flatnonzero(hit & layout.boundary)[0])]['observation.state'])[CI.GOAL_Y_DIM]:+.4f} "
          f"→ 供体 {layout.donor_vals[int(np.flatnonzero(hit & layout.boundary)[0])][0]:+.4f}")

    print("\n" + ("全部通过" if OK else "有 FAIL，见上"))
    if not OK:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
