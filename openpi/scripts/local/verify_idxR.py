#!/usr/bin/env python
"""起训前的最后一道闸：**直接从 config.py 读 idxR 那一档**，核对
  ① loss_weights 长度与右爪那两个下标，② 相位权重真的生效、成分如设计。
权重不手抄 —— 手抄的副本会跟配置漂移，这正是"改了权重却训了旧档"的经典事故。

必须写成 .py 跑（stdin/heredoc 会让 lerobot 的 spawn worker 找不到 <stdin>）。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_idxR.py
"""
from __future__ import annotations

import numpy as np

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.training import cycle_index as CI
from openpi.training import config as _config

CONFIG_NAME = "pi05_g1_omnipicker_clear5_d22idxR_lora_12k"
WANT_PHASE = ((0, 3.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0))
# 上一档 idxP 的相位权重 —— 本轮相对它的 delta 才是"这轮改了什么"，别拿 idxI 当基线。
IDXP_PHASE = ((0, 3.0), (1, 2.0), (4, 2.0))
WANT_LW = [0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10
TOOL_ORDER = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
OK = True


def check(cond: bool, msg: str) -> None:
    global OK
    OK = OK and bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")


def main() -> None:
    cfg = _config.get_config(CONFIG_NAME)
    print(f"### 0) 从 config.py 读到的 {CONFIG_NAME}")
    print(f"  exp_name     {cfg.exp_name}")
    print(f"  num_train_steps {cfg.num_train_steps}")
    print(f"  action_dim   {cfg.model.action_dim}")
    print(f"  weight_loader {type(cfg.weight_loader).__name__}")
    src = getattr(cfg.weight_loader, "params_path", None) or getattr(cfg.weight_loader, "_path", None)
    print(f"  热启动路径   {src}")
    check("5000" in str(src), "热启动自 idxP/5000")

    # ---------------- 1) loss_weights ----------------
    print("\n### 1) loss_weights（32 维；d22 布局 0-6 左臂 / 7-13 右臂 / 14-15 左爪 / 16-17 右爪 / 18 距离 / 19-21 通道）")
    lw = list(cfg.model.loss_weights)
    check(len(lw) == 32, f"长度 {len(lw)} == 32")
    check(len(lw) == cfg.model.action_dim, f"长度 == action_dim({cfg.model.action_dim})")
    check(tuple(lw) == tuple(WANT_LW), "逐元素 == 本轮设计")
    names = (
        [f"L臂{i}" for i in range(7)]
        + [f"R臂{i}" for i in range(7)]
        + ["L爪14", "L爪15"]
        + ["R爪16", "R爪17"]
        + ["距离18", "goal_y19", "cyc_dist20"]
        + ["cyc_idx21"]
        + [f"pad{i}" for i in range(22, 32)]
    )
    tot = sum(lw)
    for i in range(16, 22):
        print(f"    [{i:>2}] {names[i]:<11} w={lw[i]:<5} 占比 {100 * lw[i] / tot:5.2f}%")
    rsum_old = sum([5.0, 5.0] + [5.0] + [3.0] * 3 + [1.0])  # idxP 的 16..21
    rsum_new = sum(lw[16:22])
    print(f"  右爪16/17: idxP 5.0 -> 本轮 {lw[16]}（爪权重占比 {100 * 10.0 / (tot - 6.0):.1f}% -> {100 * 16.0 / (tot - 6.0 + 6.0):.1f}%）")
    check(lw[16] == 8.0 and lw[17] == 8.0, "右爪 16/17 == 8.0")
    check(lw[14] == 1.0 and lw[15] == 1.0, "左爪 14/15 保持 1.0（它常数0.048，别去动）")
    check(all(v == 0.05 for v in lw[:7]), "左臂 0.05 保持不变")
    check(lw[18] == 5.0, "距离 18 保持 5.0（单变量：这轮不动它）")

    # ---------------- 2) 相位权重 ----------------
    pw = tuple(cfg.data.base_config.cycle_phase_weights)
    print(f"\n### 2) cycle_phase_weights = {pw}")
    check(pw == WANT_PHASE, "== 本轮设计 (0:3 1:2 2:2 3:2 4:2)")
    print(f"  cycle_boundary_window={cfg.data.base_config.cycle_boundary_window} "
          f"weight={cfg.data.base_config.cycle_boundary_weight} "
          f"cf_prob={cfg.data.base_config.cycle_counterfactual_prob}")

    # ---------------- 3) 真实采样器数成分 ----------------
    # **基线必须是 idxP，不是 idxI**：本轮是接 idxP/5000 热启动的，"多了多少梯度"要比上一档。
    # 更要紧的是**用绝对样本数，别用占比%**：p0 加权 3x 会把总池撑大 1.49 倍，于是每个相位的
    # 占比都被稀释，p2/p3 的占比反而从 19.6% "掉"到 18.8%，而绝对量其实涨了 42.6%。
    # 模型是按 base 均匀洗牌取 batch 的，某相位拿到多少梯度 ∝ count_p 的**绝对值**。
    print("\n### 3) 成分（用采样器实际吐出的索引数一遍；绝对样本数才是梯度占比）")
    repo = cfg.data.repo_id
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo)
    ds = lerobot_dataset.LeRobotDataset(repo, delta_timestamps={"action": [t / meta.fps for t in range(20)]})
    layout = CI.build_cycle_layout(ds, window=cfg.data.base_config.cycle_boundary_window, seed=0)
    n = len(layout)
    n_ph = max(layout.phase_sizes()) + 1
    sizes = layout.phase_sizes()
    # 边界帧本来就含在 arange(n) 里，只被**额外**复制 extra_b 遍；相位帧同理额外复制 extra_p 遍。
    # ⇒ count_p = n_p*(1+extra_p) + extra_b*bdup_p。实测逐相位对上（校验见下）。
    extra_b = max(int(round(cfg.data.base_config.cycle_boundary_weight)) - 1, 0)
    bdup = np.bincount(layout.phase[layout.boundary], minlength=n_ph)

    def comp(weights: dict) -> tuple[np.ndarray, int]:
        smp = CI.CyclePhaseSampler(
            layout,
            boundary_weight=cfg.data.base_config.cycle_boundary_weight,
            phase_weights=weights,
            seed=0,
        )
        idx = np.array(list(iter(smp)), dtype=np.int64)
        return np.bincount(layout.phase[idx], minlength=n_ph), len(idx)

    prev_cnt, prev_n = comp(dict(IDXP_PHASE))  # 上一档 idxP = 我们接续的起点
    new_cnt, new_n = comp(dict(pw))

    hdr = f"  {'相位':<14}{'idxP绝对':>10}{'idxR绝对':>10}{'倍数':>9}   {'idxP占比':>9}{'idxR占比':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in range(n_ph):
        who = TOOL_ORDER[p] if p < len(TOOL_ORDER) else "收尾段(第4次开爪后)"
        print(
            f"  p{p} {who:<11}{prev_cnt[p]:>10}{new_cnt[p]:>10}{new_cnt[p] / prev_cnt[p]:>8.2f}x"
            f"   {100 * prev_cnt[p] / prev_n:>8.1f}%{100 * new_cnt[p] / new_n:>8.1f}%"
        )
    print(f"  {'合计':<14}{prev_n:>10}{new_n:>10}{new_n / prev_n:>8.2f}x")

    # 计数公式对账（防止哪天采样器的复制语义改了而校验器还在自说自话）
    for p, w in pw:
        pred = int(sizes[p] * (1 + (round(w) - 1)) + extra_b * bdup[p])
        check(pred == new_cnt[p], f"p{p} 计数 == n_p*(1+extra_p)+{extra_b}*bdup（预测 {pred}）")

    # 本轮设计的**唯一**意图：把 p2/p3 从 idxP 的低位抬到与 p1/p4 齐平，其余一格不动。
    print("\n  相对 idxP 的实际改动（单变量核查）：")
    for p in range(n_ph):
        r = new_cnt[p] / prev_cnt[p]
        who = TOOL_ORDER[p] if p < len(TOOL_ORDER) else "收尾段"
        tag = "← 本轮改动" if p in (2, 3) else "未动"
        print(f"    p{p} {who:<10} {r:.3f}x  {tag}")
    for p in (2, 3):
        check(
            new_cnt[p] > prev_cnt[p],
            f"p{p}({TOOL_ORDER[p]}) 绝对样本被抬起（{prev_cnt[p]} -> {new_cnt[p]}，"
            f"{new_cnt[p] / prev_cnt[p]:.3f}x）",
        )
    check(new_cnt[2] == new_cnt[3], f"p2 == p3（{new_cnt[2]} == {new_cnt[3]}）")
    check(new_cnt[2] == new_cnt[1] == new_cnt[4], "p2/p3 已与 p1/p4 齐平（四者相等）")
    for p in (0, 1, 4, 5):
        check(
            new_cnt[p] == prev_cnt[p],
            f"p{p}({TOOL_ORDER[p] if p < 5 else '收尾段'}) 与 idxP 逐样本相同（{prev_cnt[p]}），未受影响",
        )
    check(abs(sum(new_cnt) - new_n) < 1e-9, "各相位样本数之和 == 每轮总样本")

    print("\n" + ("全部通过 —— 可以起训" if OK else "**有 FAIL，别拿这档去训**"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
