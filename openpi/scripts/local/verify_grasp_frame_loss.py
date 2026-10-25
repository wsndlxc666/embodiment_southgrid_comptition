#!/usr/bin/env python
"""复核"合爪帧加权"这一项 loss 公式改动：**关掉时逐位不变、打开时只打在该打的地方**。

为什么单独写一个复核脚本：本仓已经三次栽在"判据/被查对象与真正生效的那条路径不是同一个
东西"上（`早上看这里_idxS.md` §2.1）。所以这里刻意分成三层查——

  第 1 层  **幂等/恒等**：`grasp_frame_weight=1.0` 时输出必须与输入**逐位相同**（`array_equal`，
           不是 `allclose`）。这一条保证所有既有配置的 loss 一个比特都没动。
  第 2 层  **落点**：打开时，被放大的格子必须**恰好**是 `{右爪闭合的帧} × grasp_dims`，
           其余格子严格等于原值（连 1e-7 的漂移都不许有）。查的是**真的那些方法体**
           （用 SimpleNamespace 把真属性绑上去调 Pi0.apply_grasp_frame_weights /
           Pi0.weight_loss_per_dim 本身，不是照抄一遍公式）。
  第 3 层  **接线**：`scripts/train.py`（真反传的那条路）和 `Pi0.compute_loss` 是否都调
           `weight_loss_per_dim`。**这一条是源码级检查**，不是执行级 —— 见文件末尾的
           "没验到什么"。

再加一段**真实数据**的测量：dim16 到底是不是右爪、闭合帧占多少。这一项要是只占 1% 的帧，
那它再大也治不了"抓不牢"，读数得先看这个。

用法（纯离线，不起仿真、不训练、不写任何文件）：
  /home/dgl/lxc/openpi/.venv/bin/python scripts/local/verify_grasp_frame_loss.py
  ... --n_ep 8          # 多抽几集
"""
from __future__ import annotations

import argparse
import glob
import inspect
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import jax.numpy as jnp  # noqa: E402

from openpi.models import pi0 as _pi0  # noqa: E402

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
AD, AH = 32, 20
GRIP, THRESH = 16, 0.5
GRASP_DIMS = (7, 8, 9, 18)

OK = "\033[32m✓\033[0m"
BAD = "\033[31m✗\033[0m"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {OK if cond else BAD} {msg}")
    if not cond:
        fails.append(msg)


def stand_in(weight: float, loss_weights=None):
    """一个只带"真属性"的空壳，用来调 Pi0 上**真的**那两个方法体。"""
    m = types.SimpleNamespace()
    m.grasp_frame_weight = float(weight)
    m.grasp_grip_dim = GRIP
    m.grasp_close_thresh = THRESH
    if weight > 1.0:
        mask = np.zeros(AD, dtype=np.float32)
        for d in GRASP_DIMS:
            mask[d] = 1.0
        m.grasp_dim_mask = tuple(mask.tolist())
    else:
        m.grasp_dim_mask = None
    if loss_weights is None:
        m.loss_weights = None
    else:
        w = np.asarray(loss_weights, dtype=np.float32)
        m.loss_weights = tuple((w / w.mean()).tolist())
    # 把**真的方法体**绑上去，这样 weight_loss_per_dim 内部那两次 self.apply_* 走的也是
    # Pi0 上真正的实现，而不是在这里照抄一遍公式（照抄就正是"假闸"）。
    m.apply_loss_weights = types.MethodType(_pi0.Pi0.apply_loss_weights, m)
    m.apply_grasp_frame_weights = types.MethodType(_pi0.Pi0.apply_grasp_frame_weights, m)
    return m


def synthetic(seed: int = 0, closed_pattern=None):
    """(per_dim, actions)：actions 的 dim16 按给定模式在 0.048 / 1.0 之间跳。"""
    rng = np.random.default_rng(seed)
    actions = rng.normal(0.0, 0.1, size=(2, AH, AD)).astype(np.float32)
    actions[..., GRIP] = 0.048
    if closed_pattern is None:
        closed_pattern = np.zeros((2, AH), dtype=bool)
        closed_pattern[0, 7:13] = True  # 0 号样本第 7..12 帧闭合
        closed_pattern[1, 15:] = True  # 1 号样本第 15..19 帧闭合
    actions[..., GRIP] = np.where(closed_pattern, 1.0, 0.048).astype(np.float32)
    per_dim = rng.uniform(0.0, 1.0, size=(2, AH, AD)).astype(np.float32)
    return per_dim, actions, closed_pattern


def layer1_identity() -> None:
    print("\n=== 第 1 层：weight=1.0 时必须**逐位**不变 ===")
    per_dim, actions, _ = synthetic()
    m = stand_in(1.0)
    out = np.asarray(_pi0.Pi0.apply_grasp_frame_weights(m, jnp.asarray(per_dim), jnp.asarray(actions)))
    check(np.array_equal(out, per_dim), "apply_grasp_frame_weights 关掉时逐位相同 (array_equal)")

    m2 = stand_in(1.0, loss_weights=[1.0] * AD)
    out2 = np.asarray(_pi0.Pi0.weight_loss_per_dim(m2, jnp.asarray(per_dim), jnp.asarray(actions)))
    check(np.array_equal(out2, per_dim), "weight_loss_per_dim 全 1 权重时逐位相同 (array_equal)")

    # 关掉 + 有逐维权重时，必须与旧公式(只乘逐维权重)逐位相同
    lw = np.array([0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10, dtype=np.float32)
    m3 = stand_in(1.0, loss_weights=lw)
    new = np.asarray(_pi0.Pi0.weight_loss_per_dim(m3, jnp.asarray(per_dim), jnp.asarray(actions)))
    old = per_dim * np.asarray(m3.loss_weights, dtype=np.float32)  # 改动前的公式
    check(np.array_equal(new, old), "关掉时 == 旧公式(per_dim × loss_weights) 逐位相同")


def layer2_placement() -> None:
    print("\n=== 第 2 层：打开时放大的格子恰好是 {闭合帧} × grasp_dims ===")
    W = 6.0
    per_dim, actions, closed = synthetic()
    m = stand_in(W)
    out = np.asarray(_pi0.Pi0.apply_grasp_frame_weights(m, jnp.asarray(per_dim), jnp.asarray(actions)))
    ratio = np.where(per_dim != 0, out / per_dim, 0.0)

    want = np.zeros((2, AH, AD), dtype=bool)
    want[:, :, list(GRASP_DIMS)] = True
    want &= closed[:, :, None]
    check(np.allclose(ratio[want], W), f"该加权的 {int(want.sum())} 个格子全部 ×{W}")
    check(np.allclose(ratio[~want], 1.0), f"其余 {int((~want).sum())} 个格子严格 ×1")
    # 关键反例：闭合帧上的**非** grasp_dims 不许动；非闭合帧上的 grasp_dims 也不许动
    check(np.allclose(ratio[:, :, 16][closed], 1.0), "闭合帧上的 dim16(爪指令本身) 不被放大")
    check(np.allclose(ratio[~closed][:, list(GRASP_DIMS)], 1.0), "未闭合帧上的 dim7/8/9/18 不被放大")
    # 逐维权重与帧权重必须叠乘，不是互相覆盖
    # want 横跨 dim7/8/9(w=4) 与 dim18(w=2)，所以期望值不能写成一个标量 —— 要逐格算。
    lw = np.array([1.0] * AD, dtype=np.float32)
    lw[7], lw[8], lw[9], lw[18] = 4.0, 4.0, 4.0, 2.0
    m2 = stand_in(W, loss_weights=lw)
    out2 = np.asarray(_pi0.Pi0.weight_loss_per_dim(m2, jnp.asarray(per_dim), jnp.asarray(actions)))
    wdim = np.asarray(m2.loss_weights, dtype=np.float32)
    gain = np.ones((2, AH, AD), dtype=np.float32)
    gain[want] = W
    check(np.array_equal(out2, per_dim * wdim * gain),
          "帧权重与逐维权重是**叠乘**(== per_dim × w_dim × gain) 且逐位相同")
    wdim_full = np.broadcast_to(wdim, per_dim.shape)
    check(np.array_equal(out2[~want], (per_dim * wdim_full)[~want]),
          "非加权格子上帧权重是恒等(退化成旧公式)")


def layer3_wiring() -> None:
    print("\n=== 第 3 层：接线（源码级，非执行级）===")
    tr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train.py")
    with open(tr, encoding="utf-8") as fh:
        train_src = fh.read()
    check("weight_loss_per_dim(per_dim, actions)" in train_src,
          "scripts/train.py 走 weight_loss_per_dim(单一入口)")
    check("apply_grasp_frame_weights" not in train_src,
          "scripts/train.py 没有自己另写一遍加权(那样会漏)")

    cl = inspect.getsource(_pi0.Pi0.compute_loss)
    check("compute_weighted_loss_per_dim" in cl, "Pi0.compute_loss 走 compute_weighted_loss_per_dim")
    wl = inspect.getsource(_pi0.Pi0.weight_loss_per_dim)
    check("apply_loss_weights" in wl and "apply_grasp_frame_weights" in wl,
          "weight_loss_per_dim 同时含逐维权重与帧权重")
    # 合爪帧加权必须在**唯一的**加权出口里，不能有第三条路绕过它
    body = inspect.getsource(_pi0.Pi0)
    check(body.count("def weight_loss_per_dim") == 1 and body.count("apply_grasp_frame_weights(") == 2,
          "加权出口唯一(定义 1 处 / 被调 2 处：weight_loss_per_dim 与 compute_weighted_loss_per_dim)")


def real_data(n_ep: int) -> None:
    print(f"\n=== 真实数据：dim16 是不是右爪、闭合帧占多少（抽 {n_ep} 集）===")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  (没有 pyarrow，跳过；用 openpi 的 .venv 跑)")
        return
    files = sorted(glob.glob(os.path.join(DS, "data", "chunk-*", "*.parquet")))[:n_ep]
    if not files:
        print(f"  (没找到 parquet：{DS})")
        return
    tot = closed_tot = 0
    per_ep = []
    for f in files:
        t = pq.read_table(f, columns=["action"])
        a = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
        g = a[:, GRIP]
        n_closed = int((g > THRESH).sum())
        tot += len(g)
        closed_tot += n_closed
        per_ep.append((os.path.basename(f), len(g), n_closed, float(g.min()), float(g.max())))
    print(f"  {'episode':<26}{'帧数':>7}{'闭合帧':>8}{'占比':>8}   dim16 min/max")
    for name, n, c, lo, hi in per_ep[:5]:
        print(f"  {name:<26}{n:>7}{c:>8}{100 * c / n:>7.1f}%   {lo:.4f} / {hi:.4f}")
    print(f"  合计：{tot} 帧，其中闭合 {closed_tot} 帧 = **{100 * closed_tot / tot:.1f}%**")
    check(all(lo < 0.06 and hi > 0.99 for _, _, _, lo, hi in per_ep),
          "dim16 确实是 0.048 张开 ↔ 1.0 闭合的方波（不是别的通道）")

    # 边界帧（张开→闭合的那一帧）才是"抓取那一刻"，单独数一下有多少
    edges = 0
    for f in files:
        a = np.asarray(pq.read_table(f, columns=["action"]).column("action").to_pylist(), dtype=np.float32)
        g = (a[:, GRIP] > THRESH).astype(np.int8)
        edges += int(((g[1:] == 1) & (g[:-1] == 0)).sum())
    print(f"  「张开→闭合」边沿（=真正的抓取那一刻）：{edges} 个 / {len(files)} 集")
    check(edges > 0, "存在抓取边沿样本")
    print("\n  读法：加权覆盖的是**闭爪之后的所有帧**（抓取+搬运+放置），不是只有抓取那一帧。")
    print("        边沿帧数 ≈ 每集抓几次；若加权只打到边沿，样本会稀疏到几乎不起作用。")
    print("        覆盖比例越高，这一项越接近「给右爪闭合期间的位置精度加一个整体增益」。")

    frac = closed_tot / tot
    print(f"\n  ⚠ 闭合占比 {100 * frac:.1f}% 意味着它**不是**外科手术式的：dim7/8/9/18 上")
    print("     的平均放大倍数 = 占比×W + (1−占比)：")
    for W in (1.5, 2.0, 3.0, 6.0):
        print(f"       W={W:<4} → 平均 ×{frac * W + (1 - frac):.2f}")
    print("     想让它更「外科」，得把窗口缩到合爪边沿附近（边沿只有 ~5 个/集，要配 ±k 帧的膨胀），")
    print("     那需要另加一个 cycle 边沿掩码 —— 本版没做。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ep", type=int, default=4)
    args = ap.parse_args()

    print(f"配置：grasp_frame_weight / grip_dim={GRIP} / thresh={THRESH} / grasp_dims={GRASP_DIMS}")
    layer1_identity()
    layer2_placement()
    layer3_wiring()
    real_data(args.n_ep)

    print("\n" + "=" * 72)
    if fails:
        print(f"{BAD} {len(fails)} 项没过：")
        for m in fails:
            print(f"    - {m}")
        sys.exit(1)
    print(f"{OK} 全过。")
    print("\n**没验到什么**（诚实说明）：")
    print("  · 没跑真实的 train_step（要数据集 + GPU + 一次前向），所以「接线」只是源码级检查。")
    print("  · 没证明这一项能让抓取变好 —— 那是训练之后才知道的事。")


if __name__ == "__main__":
    main()
