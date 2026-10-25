#!/usr/bin/env python
"""校验"按相位加权"这一档：相位数组切得对不对，采样器倍数是否符合设计，
以及 **phase_weights 为空时必须与旧的 CycleBoundarySampler 逐元素相同**。

最后那条是最要紧的：它保证加了相位加权之后，旧档（pilot / d22idxI）的 batch 成分
一点没变，否则"这轮结果是加权带来的还是采样器换了"就分不清了。

必须写成 .py 文件跑 —— stdin/heredoc 会让 DataLoader 的 spawn worker 找不到 `<stdin>`。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_phase_weight.py
"""
from __future__ import annotations

import numpy as np

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.training import cycle_index as CI

REPO = "g1_omnipicker_clear5_d22_100ep"
WINDOW = 60
BOUNDARY_WEIGHT = 10.0
# 用户要的三轮：第 1、2、5 件工具（扳手 / 螺丝刀 / 电工刀(右)）
PHASE_WEIGHTS = {0: 5.0, 1: 5.0, 4: 3.0}
# **本档实际选用的权重**，理由见 3b 那行：现状不加权时相位 0（扳手，必须第一个做的那件）
# 每轮只占 7.6%，是六个相位里最低的 —— 因为边界过采样（x10）落在相位 1..5 的开头，
# 而帧 0 那段刻意不算边界，于是相位 0 一点没被抬。也就是说 pilot 那套配置**无意中**
# 给"第一件工具"造了 2.6x 的样本赤字。0:3 先把这个赤字补平，再谈额外强化。
CHOICE = {0: 3.0, 1: 2.0, 4: 2.0}
TOOL_ORDER = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
OK = True


def check(cond: bool, msg: str) -> None:
    global OK
    OK = OK and bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")


def main() -> None:
    meta = lerobot_dataset.LeRobotDatasetMetadata(REPO)
    ds = lerobot_dataset.LeRobotDataset(
        REPO, delta_timestamps={"action": [t / meta.fps for t in range(20)]}
    )
    print(f"数据集 {REPO}: {len(ds)} 帧, {meta.total_episodes} 集")

    layout = CI.build_cycle_layout(ds, window=WINDOW, seed=0)
    n = len(layout)
    cps = layout.change_points[0]
    n_ph = max(layout.phase_sizes()) + 1

    # ---------------- 1) 相位数组 ----------------
    print("\n### 1) 相位切分（相位 p = 第 p 件工具被取走的整轮）")
    sizes = layout.phase_sizes()
    for p in sorted(sizes):
        if p < len(TOOL_ORDER):
            who = f"第 {p + 1} 件 {TOOL_ORDER[p]}"
        else:
            who = "收尾段（第 4 次开爪之后，target 冻结）"
        print(f"  相位 {p} {who:<34} {sizes[p]:>7} 帧  {100 * sizes[p] / n:5.1f}%")
    check(sum(sizes.values()) == n, f"各相位帧数之和 {sum(sizes.values())} == 全局 {n}")

    # ep0 逐相位边界应当正好是 [0]+变化点 切出来的区间
    ph0 = layout.phase[layout.ep == 0]
    starts = [0, *cps]
    ends = [*cps, len(ph0)]
    want = np.zeros(len(ph0), dtype=np.int32)
    for p, (s, e) in enumerate(zip(starts, ends)):
        want[s:e] = p
    check(np.array_equal(ph0, want), f"ep0 相位号 == [0]+变化点 切出的区间，边界 {cps}")
    check(
        all(np.array_equal(layout.phase[layout.ep == e], want) for e in range(0, layout.n_episodes, 17)),
        "抽样 6 集相位号与 ep0 逐元素一致（数据是逐字节相同的脚本）",
    )
    check(n_ph == 6, f"共 {n_ph} 个相位（5 轮 + 收尾段）")

    # ---------------- 2) 与旧采样器逐元素相同 ----------------
    print("\n### 2) phase_weights 为空时 == 旧 CycleBoundarySampler（旧档 batch 成分不变）")
    new = CI.CyclePhaseSampler(layout, boundary_weight=BOUNDARY_WEIGHT, phase_weights={}, seed=0)
    old = CI.CycleBoundarySampler(layout, weight=BOUNDARY_WEIGHT, seed=0)
    check(len(new) == len(old), f"每轮样本数 {len(new)} == {len(old)}")
    it_new, it_old = iter(new), iter(old)
    ident = all(a == b for a, b in zip(it_new, it_old))
    check(ident, "第 1 个 epoch 的索引序列逐元素相同")
    # 第 2 个 epoch 也要相同（两边都用 seed+epoch 派生，容易在这里写歪）
    it_new, it_old = iter(new), iter(old)
    ident2 = all(a == b for a, b in zip(it_new, it_old))
    check(ident2, "第 2 个 epoch 的索引序列逐元素相同")
    # 同一采样器两次迭代必须不同（打乱确实在变），否则上面那条平凡成立
    a1 = list(iter(CI.CyclePhaseSampler(layout, boundary_weight=BOUNDARY_WEIGHT, seed=0)))
    check(not np.array_equal(np.array(a1), np.array(list(iter(new)))), "跨 epoch 顺序确实在变（非平凡）")

    # ---------------- 3) 加权后的成分 ----------------
    print(f"\n### 3) 加相位权重 {PHASE_WEIGHTS} 后的每轮成分")
    s = CI.CyclePhaseSampler(
        layout, boundary_weight=BOUNDARY_WEIGHT, phase_weights=PHASE_WEIGHTS, seed=0
    )
    idx = np.array(list(iter(s)), dtype=np.int64)
    got_ph = np.bincount(layout.phase[idx], minlength=n_ph)
    # 设计值：相位 p 的每轮出现次数 = size(p) × round(w_p)（边界倍数落在相位开头，
    # 边界帧会额外被复制 extra_b 次，所以这里用 >= 而不是 ==）
    for p in sorted(PHASE_WEIGHTS):
        mult = int(round(PHASE_WEIGHTS[p]))
        base_n = sizes[p] * mult
        n_b = int(layout.boundary[idx][layout.phase[idx] == p].sum())
        print(
            f"  相位 {p}: 设计 >= {base_n:>7}  实际 {got_ph[p]:>7}  "
            f"（其中 {n_b} 个是边界帧，另被边界倍数复制过）"
        )
        check(got_ph[p] >= base_n, f"相位 {p} 出现次数 >= size x {mult}")
    # 未加权的相位**不是**恰好 size 次：相位 1..5 的开头就是边界窗口，会被边界倍数
    # 再复制 extra_b 遍。（相位 0 没有边界帧 —— include_episode_start 默认 False，
    # 帧 0 那段 home 位姿刻意不算边界，所以它的首个"变化点窗口"落在相位 1 里。）
    extra_b = max(int(round(BOUNDARY_WEIGHT)) - 1, 0)
    nb_by_phase = np.bincount(layout.phase[layout.boundary], minlength=n_ph)
    for p in sorted(set(sizes) - set(PHASE_WEIGHTS)):
        want_n = sizes[p] + int(nb_by_phase[p]) * extra_b
        check(
            got_ph[p] == want_n,
            f"未加权的相位 {p}: {got_ph[p]} == {sizes[p]} + 边界 {nb_by_phase[p]}x{extra_b}",
        )
    check(int(nb_by_phase[0]) == 0, "相位 0 确实没有边界帧（帧 0 不算边界）")
    print(f"  每轮总样本 {len(s)}（加权前 {n}，放大 {len(s) / n:.2f}x）")
    print("\n" + s.describe())

    # ---------------- 3b) 候选权重并排比 ----------------
    print("\n### 3b) 候选设置对成分的影响（每轮总样本 / 各相位占比%）")
    cands = [
        ("不加权", {}),
        ("温和 0:2 1:2 4:2", {0: 2.0, 1: 2.0, 4: 2.0}),
        ("中  0:4 1:3 4:2", {0: 4.0, 1: 3.0, 4: 2.0}),
        ("强  0:5 1:5 4:3", PHASE_WEIGHTS),
        ("只 0:5", {0: 5.0}),
        ("**选定** 0:3 1:2 4:2", CHOICE),
    ]
    hdr = f"  {'设置':<20} {'每轮样本':>9} |" + "".join(f"{f'p{p}':>6}" for p in range(n_ph))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, w in cands:
        smp = CI.CyclePhaseSampler(
            layout, boundary_weight=BOUNDARY_WEIGHT, phase_weights=w, seed=0
        )
        ii = np.array(list(iter(smp)), dtype=np.int64)
        cnt = np.bincount(layout.phase[ii], minlength=n_ph)
        pct = 100.0 * cnt / len(ii)
        print(f"  {name:<20} {len(ii):>9} |" + "".join(f"{v:>5.1f}%" for v in pct))
    print("  注：p2/p3 是当前唯一做对的两件（电工刀(左)、手电筒），压低它们的占比就是")
    print("      在拿已经会的东西换还不会的东西 —— 这是本档最大的风险，选权重时盯这一行。")

    # ---------------- 4) 非法相位要报错 ----------------
    print("\n### 4) 边界情况")
    try:
        CI.CyclePhaseSampler(layout, phase_weights={99: 2.0})
        check(False, "不存在的相位应当报错")
    except ValueError as exc:
        check(True, f"不存在的相位报错: {str(exc)[:60]}")
    w1 = CI.CyclePhaseSampler(layout, phase_weights={0: 1.0})
    check(len(w1) == n, f"权重 1.0 等于不加权（{len(w1)} == {n}）")

    print("\n" + ("全部通过" if OK else "**有 FAIL，别拿这档去训**"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
