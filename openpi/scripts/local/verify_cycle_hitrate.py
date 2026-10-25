#!/usr/bin/env python
"""直接量反事实命中率：记录 loader 真正**抽到**的索引 + `_hit` 的**真实返回值**。

起因：`verify_cycle_loader_e2e.py` 实测 38.75%（state[19] 与 action[0][19] 不一致的比例），
而按设计推算只有 28.16%。差 10 个百分点有两种可能，必须分开：
  (a) 只是 160 个样本的抽样波动 —— 没问题，实际干预比设计还多；
  (b) `_hit` 在**非边界帧**上也返回 True —— 那是在污染本该干净的样本，是 bug。

第一版这个脚本用「重建一份 layout + 复算 rng」来判定，结果报出「非边界帧命中 23/55」。
但 `_hit` 的第一行就是 `if not self._layout.boundary[i]: return False`，
**非边界命中按构造不可能** ⇒ 错的是那份复算，不是训练代码：
  · 重建的 layout 和 loader 用的不是同一个对象；和/或
  · `create_data_loader` 的 seed 默认值不等于 `cfg.seed`，复算的 rng 对不上。
所以这一版改成**直接记录 `_hit` 的返回值**，并且从 self 上取 layout，不再有任何复算。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_cycle_hitrate.py
"""
from __future__ import annotations

import dataclasses

import numpy as np

import openpi.training.config as _config
import openpi.training.cycle_index as _ci
import openpi.training.data_loader as _dl

CONFIG = "pi05_g1_omnipicker_clear5_d22idx_lora_20k"
N_BATCHES = 40


def main() -> None:
    asked: list[int] = []
    answered: list[bool] = []
    ref: dict[str, object] = {}
    orig = _ci.CycleCounterfactualDataset._hit

    def spy(self, i):  # noqa: ANN001, ANN202
        r = orig(self, i)
        ref.setdefault("layout", self._layout)
        ref.setdefault("seed", self._seed)
        ref.setdefault("prob", self._prob)
        asked.append(int(i))
        answered.append(bool(r))
        return r

    _ci.CycleCounterfactualDataset._hit = spy
    try:
        cfg = _config.get_config(CONFIG)
        cfg = dataclasses.replace(cfg, num_workers=0)
        loader = _dl.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=N_BATCHES)
        n = 0
        for batch in loader:
            obs, _ = batch
            n += np.asarray(obs.state).shape[0]
    finally:
        _ci.CycleCounterfactualDataset._hit = orig

    idx = np.asarray(asked)
    hit = np.asarray(answered)
    layout = ref["layout"]
    print(f"\nloader 产出 {n} 个样本，_hit 被问了 {len(idx)} 次")
    print(f"  loader 实际用的：seed={ref['seed']}  prob={ref['prob']}  "
          f"layout 边界帧 {layout.n_boundary}/{len(layout)}")
    if len(idx) == 0:
        print("✗ _hit 一次都没被调用 —— 反事实那层根本没接上")
        return

    # 边界掩码从 **同一个 layout 对象** 上取，不再重建
    b = np.asarray(layout.boundary)[idx]
    nb, n_nb = int(b.sum()), int((~b).sum())
    print(f"\n  抽到的索引里：边界帧 {nb}/{len(idx)} = {100*b.mean():.2f}%"
          f"   非边界 {n_nb}   唯一索引 {len(np.unique(idx))}")
    print(f"  边界帧上 _hit 命中   {int(hit[b].sum())}/{nb} = "
          f"{100*hit[b].mean() if nb else float('nan'):.2f}%   (设计 p={ref['prob']})")
    print(f"  **非边界帧上 _hit 命中 {int(hit[~b].sum())}/{n_nb}**"
          f"   ← 按构造必须是 0；不为 0 才说明训练代码在污染干净样本")
    print(f"  **总命中率 = {100*hit.mean():.2f}%**")
    print(f"  设计预期总命中率 = "
          f"{100 * (layout.n_boundary * max(round(10.0), 1) / (len(layout) + (max(round(10.0), 1) - 1) * layout.n_boundary)) * ref['prob']:.2f}%"
          f"   （边界占比 {100*layout.n_boundary/len(layout):.2f}% → 采样后 "
          f"{100 * layout.n_boundary * 10 / (len(layout) + 9 * layout.n_boundary):.2f}%）")
    design = (layout.n_boundary * 10 / (len(layout) + 9 * layout.n_boundary)) * ref["prob"]
    se = np.sqrt(len(hit) * design * (1 - design))
    print(f"  实测 vs 设计：{hit.sum()} vs {len(hit)*design:.1f} 个样本，"
          f"二项 sd={se:.1f} ⇒ {(hit.sum() - len(hit)*design)/max(se,1e-9):+.2f}σ")


if __name__ == "__main__":
    main()
