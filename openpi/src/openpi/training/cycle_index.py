"""按「周期相位」重加权 / 反事实增强 —— 专门给 d22 的 goal_y/cycle_dist/cycle_idx 用。

背景（全部实测，只读数据集，见 scripts/local/ 下四个脚本）：

  `phase_pose_diag.py`：4 个真正变化点(366/766/1166/1566)上的右末端位置云心彼此只差
  **8~18mm**，而各云自身散布 1~46mm —— 位置在边界处**分不出相位**。同时 goal_y 逐集
  std=0.0000（100 集完全一致），相邻相位差 77~131mm。

  `phase_target_conflict.py`（判决实验，不需要模型）：变化点后 40 帧内，**65~68%** 的样本
  能在别的相位里找到位置差 <10mm 的邻居；这些样本未来 20 帧的真值轨迹相差**中位 44mm**
  （p90 76mm），而"同相位、同样近"的抖动地板只有 **6mm** —— 差 7~10 倍。
  ⇒ 「输入几乎相同、输出差很远」的样本大量存在，索引通道的梯度是真实且充足的。

  那为什么 cycle_idx 的权柄这么小？**因为冗余**：能定相位的不止 cycle_idx ——
  dim19 goal_y 本身就是逐集完全一致的纯净相位标签，就躺在 state 文字串里；
  头相机还能直接看到"桌上还剩哪几件工具"。模型读任一个就够了。

  **只换 dims19/20 挡不住冗余（实测无效）**：`clear5_d22idx` 那轮只把 dims19/20 换成
  异相位真值、图像不动，结果 action dim19 的 loss 掉了 191 倍、dim21 掉了 108 倍 ——
  模型确实学会了在 index 通道内部解开 (19,20)↔21 的矛盾，**但手臂没跟着 idx 走**。
  原因是："读图"和"读 idx"给出**同一个 action**（动作空间里没有任何分量是只有 idx
  能解释的）⇒ 指向 idx 的**相对梯度是零**，再多步数也训不出来。

  所以 `CycleCounterfactualDataset` 还有一个 `swap_images` 档：命中时**把两路图像也换成
  供体（异相位、同相位内偏移）那一帧的**，只有 dim21 和 action 真值来自参照帧。
  于是「图像说第 q 轮、idx 说第 p 轮、真值动作是第 p 轮的」，**idx 成了唯一解释** ——
  这才是那条本来为 0 的梯度。注意图像与臂位姿是**故意矛盾**的（臂位姿仍是参照帧的，
  因为 action 真值必须是参照帧的、而 action[t] == state[t+1]），这是"教会它以 idx 为准"
  的对抗性输入，不是数据损坏；prob 只取一半就是为了不让它彻底不信视觉。

所以这里做三件事，都**只在变化点之后 window 帧内**生效：

  1. `CycleBoundarySampler` —— 把这段窗口过采样 weight 倍。
     只是改变每个 batch 的**成分**，`num_train_steps` 不动，总步数不变。
  2. `CycleCounterfactualDataset`（基础档）—— 以 prob 概率把这些样本观测里的 dims 19/20
     (goal_y / cycle_dist) 换成**同一集另一个相位**的真实值（不是填零、不是加噪，
     值域完全在分布内），dim21 cycle_idx 保持正确。单独用**已证明不够**，见上。
  3. `swap_images=True`（改法 A）—— 第 2 条之上再把图像也换成供体帧的，让 idx 成为
     相位身份的唯一来源。

三个组件共用同一份 `CycleLayout`（全局索引 → 集/帧、边界掩码、反事实供体帧与供体值），
所以采样和增强作用的**永远是同一批帧**。

开关全在 DataConfig 上，默认全关（0/1.0/0.0）⇒ 关掉即退化成原来的 shuffle。
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import logging
from typing import SupportsIndex

import numpy as np
import torch

# d22 的四个"周期"维在 state/action 里的下标
GOAL_DIST_DIM = 18
GOAL_Y_DIM = 19
CYCLE_DIST_DIM = 20
CYCLE_IDX_DIM = 21

# 反事实增强替换的观测维：goal_y + cycle_dist（相位身份就藏在这两维里）
CF_DIMS = (GOAL_Y_DIM, CYCLE_DIST_DIM)


def find_lerobot_dataset(dataset: object, _depth: int = 0) -> object:
    """从 TransformedDataset / PromptFromLeRobotTask 这几层包装里挖出 LeRobotDataset。"""
    if hasattr(dataset, "episode_data_index"):
        return dataset
    inner = getattr(dataset, "_dataset", None)
    if inner is None or _depth > 8:
        raise TypeError(
            f"在 {type(dataset).__name__} 里找不到 LeRobotDataset（没有 episode_data_index，"
            f"也没有可下钻的 _dataset）。cycle_index 只支持 LeRobot 数据集，"
            f"rlds/droid 那条路请把 cycle_boundary_window 留 0。"
        )
    return find_lerobot_dataset(inner, _depth + 1)


def _load_states(lr: object) -> np.ndarray:
    """(N, D) 的 observation.state。优先走 numpy 格式，避免 210k×22 个 Python 对象。"""
    try:
        col = lr.hf_dataset.with_format("numpy")["observation.state"]
        arr = np.asarray(col, dtype=np.float64)
        if arr.ndim == 2:
            return arr
    except Exception as exc:  # noqa: BLE001 - 换条路而已
        logging.debug("with_format('numpy') 取 state 失败(%s)，回落到逐元素转换", exc)
    return np.stack(
        [np.asarray(v, dtype=np.float64) for v in lr.hf_dataset["observation.state"]]
    )


@dataclasses.dataclass(frozen=True)
class CycleLayout:
    """全局样本索引 ↔ (集, 帧)，相位边界掩码，以及反事实供体。"""

    ep: np.ndarray  # (N,) int32
    frame: np.ndarray  # (N,) int32
    boundary: np.ndarray  # (N,) bool —— 变化点之后 window 帧
    donor_vals: np.ndarray  # (N, len(CF_DIMS)) float32 —— 供体帧的 dims 19/20；非边界为 0
    donor_frame: np.ndarray  # (N,) int64 —— 供体帧的**全局索引**（换图用）；非边界指向自己
    n_episodes: int
    change_points: list[list[int]]  # 每集的变化点帧号（不含帧 0），只作日志/校验
    # (N,) int32 —— 该帧属于第几个相位（0 基，按集内 cycle_idx 的变化点切）。
    # 相位 p = 第 p 件工具(TOOL_REF[p])被取走的**一整轮**：接近 → 下探 → 闭爪 → 抬起 →
    # 搬到盒子 → 开爪放下，直到下一次闭爪前。所以 p=0 涵盖"五件工具都在场"的那一段。
    phase: np.ndarray

    @property
    def n_boundary(self) -> int:
        return int(self.boundary.sum())

    def phase_sizes(self) -> dict[int, int]:
        """每个相位有多少帧（用来核对加权前的成分）。"""
        vals, cnts = np.unique(self.phase, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    def __len__(self) -> int:
        return int(self.ep.shape[0])


def build_cycle_layout(
    dataset: object,
    *,
    window: int,
    seed: int = 0,
    include_episode_start: bool = False,
) -> CycleLayout:
    """扫一遍数据集的 cycle_idx，切出变化点后的窗口，并给每个边界帧配一个异相位供体。

    window: 变化点之后多少帧算"边界窗口"。
      phase_target_conflict.py 实测：k=40 时跨相位邻居占比 67.5%、真值差中位 44mm；
      k=60 降到 41%（因为轨迹已经开始分开）、但真值差升到 65.9mm。取 60 覆盖整段。
    include_episode_start: 是否把帧 0 也算边界。默认 False —— 帧 0 是 home 位姿，
      离其它相位 660mm，本来就不歧义（phase_pose_diag.py），拉进来只是稀释。
    """
    lr = find_lerobot_dataset(dataset)
    d = lr.episode_data_index
    ep_from = np.asarray(d["from"], dtype=np.int64)
    ep_to = np.asarray(d["to"], dtype=np.int64)
    n_ep = len(ep_from)
    n = int(ep_to[-1])

    ep = np.empty(n, dtype=np.int32)
    frame = np.empty(n, dtype=np.int32)
    for e, (a, b) in enumerate(zip(ep_from, ep_to)):
        ep[a:b] = e
        frame[a:b] = np.arange(b - a, dtype=np.int32)

    states = _load_states(lr)
    if states.shape[0] != n:
        raise ValueError(f"state 行数 {states.shape[0]} != episode_data_index 总数 {n}")
    ci = states[:, CYCLE_IDX_DIM]

    boundary = np.zeros(n, dtype=bool)
    phase = np.zeros(n, dtype=np.int32)
    donor_vals = np.zeros((n, len(CF_DIMS)), dtype=np.float32)
    # 非边界帧指向自己 —— 万一有人在非边界上误触发，换图是个无害的恒等操作，
    # 而不是把帧 0 的图像搬过来。
    donor_frame = np.arange(n, dtype=np.int64)
    cps: list[list[int]] = []
    rng = np.random.default_rng(seed)

    for e, (a, b) in enumerate(zip(ep_from, ep_to)):
        seq = ci[a:b]
        chg = (np.flatnonzero(seq[1:] != seq[:-1]) + 1).tolist()
        cps.append(chg)
        # 相位首帧 = [0] + chg；相位 p 的长度 = starts[p+1]-starts[p]
        starts = [0, *chg]
        ends = [*chg, int(b - a)]
        # 逐帧打相位号（相位 p 覆盖 [starts[p], ends[p])）。这一段与 boundary/供体的
        # 取舍无关 —— 某相位找不到供体时下面的边界掩码会被撤回，但相位号照常成立，
        # 所以"按相位加权"不会被供体的可得性牵连。
        for p, (s0, s1) in enumerate(zip(starts, ends)):
            phase[a + s0 : a + s1] = p
        # 每个相位自己的 goal_y。相位内恒定，但**相位 5 和相位 6 是同一个值** —— 实测
        # idx 从 5 跳到 6 时 dim19 不变（它停在最后一件工具上）。所以只按"相位不同"挑供体，
        # 会挑到 goal_y 一模一样的帧，dim 反事实退化成空操作（只有图像被换）。
        # 这里直接按 goal_y 去重，比按相位号去重更贴合"换一个目标工具"的本意。
        phase_goal_y = [float(states[a + starts[q]][GOAL_Y_DIM]) for q in range(len(starts))]
        first = 0 if include_episode_start else 1
        for p in range(first, len(starts)):
            s = starts[p]
            t = min(s + window, ends[p])
            if t <= s:
                continue
            boundary[a + s : a + t] = True
            others = [
                q for q in range(len(starts)) if q != p and phase_goal_y[q] != phase_goal_y[p]
            ]
            if not others:
                # 该集这一相位找不到"目标工具不同"的供体 —— 宁可不做，也不要空操作
                logging.debug("集 %d 相位 %d 没有异 goal_y 的供体，跳过", e, p)
                boundary[a + s : a + t] = False
                continue
            for f in range(s, t):
                q = others[int(rng.integers(len(others)))]
                # 供体保持"同相位内偏移"，这样 goal_dist/cycle_dist 的量级仍然合理，
                # 手臂也处在同一个动作阶段（都在接近/抓取），换过来的图像才不是怪姿势。
                off = min(f - s, max(ends[q] - starts[q] - 1, 0))
                j = a + starts[q] + off
                donor_vals[a + f] = states[j][list(CF_DIMS)]
                donor_frame[a + f] = j

    return CycleLayout(
        ep=ep,
        frame=frame,
        boundary=boundary,
        donor_vals=donor_vals,
        donor_frame=donor_frame,
        n_episodes=n_ep,
        change_points=cps,
        phase=phase,
    )


class CyclePhaseSampler(torch.utils.data.Sampler[int]):
    """按**相位**过采样：指定相位的帧按倍数复制，其余帧每轮一次。

    与 CycleBoundarySampler 同一套做法 —— 整数倍复制 + 可复现打乱，**不是按概率抽样**：
    每个 epoch 里每个样本出现的次数精确可控。`phase_weights` 为空时与
    CycleBoundarySampler 在同 seed 下产出**逐元素相同**的索引序列（base 的构造顺序一致）。

    相位的语义见 CycleLayout.phase：相位 p 是"第 p 件工具被取走的那一整轮"。所以
    `{0: 5.0}` 就是把"五件工具都在场 → 抓走第一件"的整段连接近带下探一起加重 5 倍。

    帧的总倍数 = 边界倍数 × 相位倍数，两个集合可交叠（边界帧位于相位开头，会同时命中），
    乘起来而不是二选一 —— 改法 A 的边界过采样不必关掉就能同时加相位权重。

    为什么用"复制"而不是给 loss 加 per-sample weight：pi0.5 的 loss 是 flow-matching
    在 batch 上取均值，采样器改的是 batch 成分，不需要动 model/loss 的签名，也不会在
    梯度累积/多进程下和权重归一化打架。代价是"权重"被量化成整数倍。
    """

    def __init__(
        self,
        layout: CycleLayout,
        *,
        boundary_weight: float = 1.0,
        phase_weights: Mapping[int, float] | None = None,
        seed: int = 0,
    ):
        self._layout = layout
        self._boundary_weight = float(boundary_weight)
        self._phase_weights = {int(k): float(v) for k, v in (phase_weights or {}).items()}
        self._seed = int(seed)
        self._epoch = 0

        present = set(layout.phase_sizes())
        missing = sorted(set(self._phase_weights) - present)
        if missing:
            raise ValueError(
                f"phase_weights 里的相位 {missing} 在数据里不存在；实际只有 {sorted(present)}"
            )

        base = np.arange(len(layout), dtype=np.int64)
        # 顺序刻意与 CycleBoundarySampler 一致：先边界、后相位。phase_weights 为空时
        # 这里一个字都不追加，于是同 seed 下与旧采样器逐元素相同（可复现性检查会验这一条）。
        extra_b = max(int(round(self._boundary_weight)) - 1, 0)
        if extra_b:
            base = np.concatenate([base, np.repeat(np.flatnonzero(layout.boundary), extra_b)])
        for p, w in sorted(self._phase_weights.items()):
            extra_p = max(int(round(w)) - 1, 0)
            if extra_p:
                base = np.concatenate(
                    [base, np.repeat(np.flatnonzero(layout.phase == p), extra_p)]
                )
        self._base = base
        if len(self._base) == 0:
            raise ValueError("采样器没有样本")

    def __len__(self) -> int:
        return int(self._base.shape[0])

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        return iter(self._base[rng.permutation(self._base.shape[0])].tolist())

    def describe(self) -> str:
        """每个相位在**一轮样本**里占多少 —— 这个数就是它拿到的梯度的比例。

        注意别只算 `n0 * mult / len(base)`：边界窗口（x boundary_weight）落在相位 1..5
        的开头，那些帧会被**额外复制** extra_b 遍，所以相位 1..5 的真实占比**高于**自己的
        加权倍数给出的值，多出来的正是边界复制。2026-09-19 实测：本档（0:3 1:2 4:2）
        只按老公式报出 p2/p3 = 6.3%，而实际是 14.8%（94000/633300）——
        差点把"压低当前唯一做对的两件"的代价读小一倍多。相位 0 没有边界帧
        （include_episode_start 默认 False，帧 0 那段 home 位姿不算边界），所以只有它
        老公式才碰巧对。这里把边界贡献单列出来，方便直接对账。
        """
        sizes = self._layout.phase_sizes()
        N = len(self._layout)
        total = len(self._base)
        extra_b = max(int(round(self._boundary_weight)) - 1, 0)
        nb_by_phase = np.bincount(self._layout.phase[self._layout.boundary], minlength=max(sizes) + 1)
        lines = [
            f"CyclePhaseSampler: 共 {N} 帧（其中边界窗口 {int(self._layout.boundary.sum())} 帧，"
            f"复制 x{self._boundary_weight:.0f}）→ 每轮 {total} 个样本"
        ]
        shown = sorted(set(sizes) | set(self._phase_weights))
        for p in shown:
            w = self._phase_weights.get(p, 1.0)
            mult = max(int(round(w)), 1)
            n0 = sizes.get(p, 0)
            n_b = int(nb_by_phase[p]) if p < len(nb_by_phase) else 0
            got = n0 * mult + n_b * extra_b
            lines.append(
                f"  相位 {p}（第 {p + 1} 件工具）: {n0:>6} 帧 {100.0 * n0 / N:5.1f}% "
                f"x{mult} + 边界 {n_b:>5}x{extra_b} = {got:>7} → 占比 {100.0 * got / total:5.1f}%"
            )
        got_all = sum(
            sizes.get(p, 0) * max(int(round(self._phase_weights.get(p, 1.0))), 1)
            + (int(nb_by_phase[p]) if p < len(nb_by_phase) else 0) * extra_b
            for p in sizes
        )
        lines.append(f"  ── 各相位合计 {got_all} / {total}" + ("  ✓ 对账一致" if got_all == total else "  ✗ 对不上！"))
        return "\n".join(lines)


class CycleBoundarySampler(torch.utils.data.Sampler[int]):
    """把边界窗口的样本按 weight 倍过采样，其余样本每轮一次。

    实现是"按整数倍复制再整体打乱"，不是按概率抽样：这样每个 epoch 里每个普通样本
    恰好出现一次、每个边界样本恰好出现 weight 次，成分精确可控、可复现。

    weight=1.0 时退化成纯随机打乱（与原来的 shuffle=True 等价），方便做对照。
    """

    def __init__(self, layout: CycleLayout, *, weight: float, seed: int = 0):
        self._layout = layout
        self._weight = float(weight)
        self._seed = int(seed)
        self._epoch = 0
        base = np.arange(len(layout), dtype=np.int64)
        extra = max(int(round(self._weight)) - 1, 0)
        if extra:
            base = np.concatenate([base, np.repeat(np.flatnonzero(layout.boundary), extra)])
        self._base = base
        if len(self._base) == 0:
            raise ValueError("采样器没有样本")

    def __len__(self) -> int:
        return int(self._base.shape[0])

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        return iter(self._base[rng.permutation(self._base.shape[0])].tolist())

    def describe(self) -> str:
        n_b = self._layout.n_boundary
        frac = n_b / len(self._layout)
        return (
            f"CycleBoundarySampler: 边界窗口 {n_b}/{len(self._layout)} = {100 * frac:.1f}% 的帧，"
            f"权重 x{self._weight:.1f} → 每轮 {len(self._base)} 个样本，"
            f"边界占比 {100 * n_b * max(int(round(self._weight)), 1) / len(self._base):.1f}%"
        )


class CycleCounterfactualDataset:
    """边界帧的 obs 换成一个"异相位"的观测，cycle_idx(21) 与 action 真值保持参照帧的。

    两档，由 swap_images 决定：
      · False（基础档，**已实测无效**）—— 只把 dims19/20(goal_y/cycle_dist) 换成异相位真值。
        模型可以照抄图像，所以指向 idx 的梯度仍然是 0。
      · True（改法 A）—— 再把两路图像换成**供体帧**的（同集、异相位、同相位内偏移）。
        此时"图像 + dim19/20 说是第 q 轮"而"dim21 与 action 真值说是第 p 轮"，
        **idx 成了唯一解释**，那条本来为 0 的相对梯度才出现。
        图像与臂位姿是**故意矛盾**的：臂位姿必须留参照帧的，因为 action[t] == state[t+1]，
        换成供体的位姿就和真值动作差了一个瞬移。

    每个全局索引的决定是**确定性**的（按索引派生随机数），所以在 DataLoader 的各个
    worker 之间不会有隐蔽的耦合，也不会因为重跑而变。
    """

    #: 观测里图像键的前缀。LeRobot 原始样本是 observation.images.<相机名>；
    #: repack 之后的 images.<相机名> 也要认，因为这个包装器的插入位置可能变。
    IMAGE_PREFIXES = ("observation.images.", "images.")

    def __init__(
        self,
        dataset: object,
        layout: CycleLayout,
        *,
        prob: float,
        seed: int = 0,
        swap_images: bool = False,
    ):
        self._dataset = dataset
        self._layout = layout
        self._prob = float(prob)
        self._seed = int(seed)
        self._swap_images = bool(swap_images)
        if self._prob > 0 and layout.n_boundary == 0:
            raise ValueError("prob>0 但边界掩码全空，检查 cycle_boundary_window / cycle_idx 那一维")
        # 图像键在这里就探明并**硬校验**：若名字不符，宁可现在 import 期就炸，
        # 也不要静默地什么都不换、白跑十小时。
        self._image_keys: tuple[str, ...] = ()
        if self._swap_images:
            probe = self._dataset[0]  # type: ignore[index]
            self._image_keys = tuple(
                k for k in probe if k.startswith(self.IMAGE_PREFIXES)
            )
            if not self._image_keys:
                raise ValueError(
                    f"swap_images=True 但在样本里找不到图像键（试过前缀 {self.IMAGE_PREFIXES}）；"
                    f"实际键：{sorted(probe)}"
                )
            logging.info("cycle 反事实换图：命中样本会替换 %s", list(self._image_keys))

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def _hit(self, i: int) -> bool:
        if self._prob <= 0:
            return False
        if not self._layout.boundary[i]:
            return False
        return float(np.random.default_rng([self._seed, i]).random()) < self._prob

    def __getitem__(self, index: SupportsIndex) -> dict:
        out = self._dataset[index]  # type: ignore[index]
        i = int(index.__index__()) if hasattr(index, "__index__") else int(index)
        if not self._hit(i):
            return out
        out = dict(out)  # 浅拷贝：下面要整键替换，别动底层缓存的字典
        vals = self._layout.donor_vals[i]
        st = out["observation.state"]
        dims = list(CF_DIMS)
        if torch.is_tensor(st):
            st = st.clone()
            st[dims] = torch.as_tensor(vals, dtype=st.dtype)
        else:
            st = np.array(st, dtype=np.float32, copy=True)
            st[dims] = vals
        out["observation.state"] = st

        if self._swap_images:
            j = int(self._layout.donor_frame[i])
            if j != i:
                donor = self._dataset[j]  # type: ignore[index]
                for k in self._image_keys:
                    out[k] = donor[k]
        return out
