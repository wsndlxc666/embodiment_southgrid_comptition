#!/usr/bin/env python
"""验证改法 A：反事实样本**真的把图像换成了异相位供体帧**，且其它一切没被误伤。

必须逐项断言，不能只看"跑起来了"：
  1. 命中的样本，两路图像 == 供体帧的图像（且供体帧与参照帧**相位不同** —— 这是全部意义）
  2. 命中的样本，dim21 cycle_idx 仍是**参照帧**的（否则就自相矛盾了，学不到东西）
  3. 命中的样本，dims19/20 == 供体帧的（和图像说同一件事）
  4. 命中的样本，臂位姿 dims0:18 与 **action 真值**逐字节不动
     （臂位姿必须留参照帧的：action[t] == state[t+1]，换位姿就和真值差一个瞬移）
  5. **未命中**的样本与原数据集逐字节相同（不能污染干净样本）
  6. 非边界帧的命中数 == 0
  7. 走**真正的 loader**（create_data_loader）再确认一次管道插对了

用法：HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data \\
      /home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_counterfactual_images.py
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

import openpi.training.config as _config
import openpi.training.cycle_index as _ci
import openpi.training.data_loader as _dl

CONFIG = "pi05_g1_omnipicker_clear5_d22idxI_lora_20k"
CFG_DIMS = list(_ci.CF_DIMS)  # [19, 20]
IDX_DIM = _ci.CYCLE_IDX_DIM  # 21
ARM_DIMS = slice(0, 18)  # 0..17：两臂 + 右爪 + goal_dist

_fail: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'✓' if ok else '✗'} {msg}")
    if not ok:
        _fail.append(msg)


def main() -> int:
    cfg = _config.get_config(CONFIG)
    model_cfg = cfg.model
    data_cfg = dataclasses.replace(cfg.data.create(cfg.assets_dirs, model_cfg))
    raw = _dl.create_torch_dataset(data_cfg, model_cfg.action_horizon, model_cfg)

    layout = _ci.build_cycle_layout(raw, window=data_cfg.cycle_boundary_window, seed=0)
    ds = _ci.CycleCounterfactualDataset(
        raw, layout, prob=1.0, seed=0, swap_images=True
    )
    states = _ci._load_states(_ci.find_lerobot_dataset(raw))

    print(f"\n布局：{layout.n_episodes} 集，边界帧 {layout.n_boundary}/{len(layout)} "
          f"= {100*layout.n_boundary/len(layout):.2f}%")
    print(f"图像键：{list(ds._image_keys)}")

    boundary_idx = np.flatnonzero(layout.boundary)
    # 取前若干个边界帧（prob=1.0 ⇒ 边界帧必命中）
    sample_idx = boundary_idx[:12].tolist()
    # 再取若干个非边界帧
    nonb = np.flatnonzero(~layout.boundary)[:8].tolist()

    print("\n【1/2/3/4】命中样本：图像/dims19-20 来自供体，dim21 与 action 来自参照")
    n_img_diff = 0
    for i in sample_idx:
        i = int(i)
        j = int(layout.donor_frame[i])
        ref = raw[i]
        don = raw[j]
        out = ds[i]
        r_st, o_st = ref["observation.state"], out["observation.state"]
        d_st = states[j]

        same_ep = int(layout.ep[i]) == int(layout.ep[j])
        diff_phase = int(states[i][IDX_DIM]) != int(states[j][IDX_DIM])
        ref_phase, don_phase = int(states[i][IDX_DIM]), int(states[j][IDX_DIM])

        img_ok = all(
            bool(np.array_equal(np.asarray(out[k]), np.asarray(don[k]))) for k in ds._image_keys
        )
        # 图像确实和参照帧不同（否则等于没换）
        img_changed = not all(
            bool(np.array_equal(np.asarray(out[k]), np.asarray(ref[k]))) for k in ds._image_keys
        )
        n_img_diff += int(img_changed)

        idx_ok = float(o_st[IDX_DIM]) == float(r_st[IDX_DIM])
        vals_ok = all(float(o_st[d]) == float(d_st[d]) for d in CFG_DIMS)
        arm_ok = bool(np.array_equal(np.asarray(out["observation.state"])[ARM_DIMS],
                                     np.asarray(r_st)[ARM_DIMS]))
        act_ok = bool(np.array_equal(np.asarray(out["action"]), np.asarray(ref["action"])))

        tag = f"i={i} ep{int(layout.ep[i])} f{int(layout.frame[i])} 相位{ref_phase}→供体j={j} 相位{don_phase}"
        if i == sample_idx[0]:
            print(f"    {tag}")
            check(same_ep, "供体与参照同集")
            check(diff_phase, "供体与参照**相位不同**（否则改了等于没改）")
            check(img_ok, "两路图像 == 供体帧图像（逐字节）")
            check(img_changed, "图像确实与参照帧不同")
            check(idx_ok, f"dim21 cycle_idx 仍是参照的（{ref_phase}）")
            check(vals_ok, "dims19/20 == 供体帧的值")
            check(arm_ok, "臂位姿 dims0:17 与参照逐字节相同")
            check(act_ok, "action 真值与参照逐字节相同")
        else:
            if not (same_ep and diff_phase and img_ok and idx_ok and vals_ok and arm_ok and act_ok):
                check(False, f"{tag} 有一项不符")

    print(f"\n【1b】{len(sample_idx)} 个命中样本里，图像真的变了的有 {n_img_diff}/{len(sample_idx)}")
    check(n_img_diff == len(sample_idx), "每个命中样本的图像都与参照帧不同")

    print("\n【5】未命中（非边界）样本应与原数据集逐字节相同")
    same = 0
    for i in nonb:
        i = int(i)
        out, ref = ds[i], raw[i]
        if layout.boundary[i]:
            continue
        eq = bool(np.array_equal(np.asarray(out["observation.state"]),
                                 np.asarray(ref["observation.state"])))
        eq &= bool(np.array_equal(np.asarray(out["action"]), np.asarray(ref["action"])))
        eq &= all(bool(np.array_equal(np.asarray(out[k]), np.asarray(ref[k]))) for k in ds._image_keys)
        same += int(eq)
    check(same == len(nonb), f"非边界样本逐个逐字节一致（{same}/{len(nonb)}）")

    print("\n【6】非边界帧的命中数必须为 0")
    hits_nb = sum(1 for i in nonb if ds._hit(int(i)))
    check(hits_nb == 0, f"非边界帧命中 {hits_nb} 个")

    print("\n【7】走真正的 loader，确认 data_loader 把 swap_images 传进来了")
    seen: dict[str, object] = {}
    orig_init = _ci.CycleCounterfactualDataset.__init__

    def spy_init(self, dataset, layout, **kw):  # noqa: ANN001, ANN202
        seen.update(kw)
        seen["image_keys"] = None
        r = orig_init(self, dataset, layout, **kw)
        seen["image_keys"] = getattr(self, "_image_keys", None)
        return r

    _ci.CycleCounterfactualDataset.__init__ = spy_init
    try:
        run_cfg = dataclasses.replace(cfg, num_workers=0)
        batch = None
        for batch in _dl.create_data_loader(
            run_cfg, framework="pytorch", shuffle=True, num_batches=20
        ):
            pass
    finally:
        _ci.CycleCounterfactualDataset.__init__ = orig_init

    # 注意：loader 产出的 state 是 (batch, 32) 而不是 22 —— PadStatesAndActions 会把 state
    # 补到模型内部的 action_dim=32（pi05 基座就是 32 维）。这是**正确**行为，
    # 别把它当成"管道断了"（第一版这里就写错成断言 22 维）。
    print(f"    最后一批 batch：state {tuple(np.asarray(batch[0].state).shape)}，"
          f"（32 = PadStatesAndActions 补的，正确）")
    print(f"    create_data_loader 传给包装器的实参：{ {k: v for k, v in seen.items() if k != 'image_keys'} }")
    check(seen.get("swap_images") is True, "create_data_loader 确实传了 swap_images=True")
    check(seen.get("prob") == 0.5, "prob 确实是配置里的 0.5")
    check(
        seen.get("image_keys") == tuple(ds._image_keys),
        f"loader 里探到的图像键与直接构造的一致：{seen.get('image_keys')}",
    )

    print()
    if _fail:
        print(f"✗ {len(_fail)} 项未通过：")
        for m in _fail:
            print(f"    - {m}")
        return 1
    print("✓ 改法 A 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
