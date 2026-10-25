#!/usr/bin/env python
"""把 clear_eval 转储里"看不到的中间步骤"重建出来，打印成一条一条，肉眼可看。

背景：转储（`--dump_chunks`）只记录了**两个边界** ——
    进去的 `state`(19维物理量) 和 出来的 `chunk`(20×18 米/四元数)。
中间那些步骤（归一化、state 离散成文字、拼 prompt、分词）**没有任何地方记录过**，
openpi 的 `serve_policy.py --record` 也只记同样这两个边界（它包在最外层）。

好消息：中间几步全是**确定性且可逆**的，所以能从转储**精确重建**（不是近似）：
    ① state 归一化 : transforms.py  _normalize_quantile
                     (x - q01)/(q99 - q01 + 1e-6)*2 - 1
    ② state 离散化 : tokenizer.py   digitize(state, linspace(-1,1,256+1)[:-1]) - 1
    ③ 拼句子      : tokenizer.py   f"Task: {prompt}, State: {state_str};\nAction: "
    ④ chunk 反推  : 把①反过来作用在 chunk 上 = 模型**原始网络输出**（不是补齐前的，
                     是截断到 18 维之后的；被丢掉的 14 个填充维无法恢复）

重建不出来的只有两样：**图像**（转储没存）和 **flow-matching 内部去噪轨迹**（50 步逐步的噪声预测）。

用法：
    python reconstruct_intermediate.py <dump.jsonl> [--query N] [--ckpt <norm_stats.json 路径>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_STATS = (
    "/home/dgl/lxc/openpi/checkpoints/pi05_g1_omnipicker_clear5_d19goal_lora_20k"
    "/clear5_d19goal_lora_100ep/9000/assets/g1_omnipicker_clear5_d19_goal_100ep/norm_stats.json"
)

# 18 维动作每一维是什么（openpi/policies/g1_omnipicker_policy.py 的约定）
ACTION_LAYOUT = [
    "l_pos_b.x", "l_pos_b.y", "l_pos_b.z",
    "l_quat_b.x", "l_quat_b.y", "l_quat_b.z", "l_quat_b.w",
    "r_pos_b.x", "r_pos_b.y", "r_pos_b.z",
    "r_quat_b.x", "r_quat_b.y", "r_quat_b.z", "r_quat_b.w",
    "l_grip_inner", "l_grip_outer",
    "r_grip_inner", "r_grip_outer",
]
STATE_EXTRA = "goal_dist"


def _normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """openpi/src/openpi/transforms.py:141-145 的逐字复刻。"""
    q01 = np.asarray(stats["q01"], dtype=np.float64)[..., : x.shape[-1]]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _denormalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize 的逆 —— 用来从归一化后的 chunk 反推模型原始输出。"""
    q01 = np.asarray(stats["q01"], dtype=np.float64)[..., : x.shape[-1]]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[..., : x.shape[-1]]
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _discretize_state(state_norm: np.ndarray) -> tuple[np.ndarray, str]:
    """openpi/src/openpi/models/tokenizer.py:24-28 的逐字复刻。"""
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    disc = np.digitize(state_norm, bins=bins) - 1
    return disc, " ".join(map(str, disc))


def _fmt_row(name: str, raw: float, norm: float, span: float) -> str:
    # span/2 = 该维归一化的物理尺度（见 memory: 分位数归一化是尺度无关的）
    phys = norm * span / 2.0
    return f"   {name:16s} 物理={raw:+10.5f}  归一化={norm:+8.4f}  该维跨度={span:9.5f}  归一化×span/2={phys:+9.6f}"


def report(path: str, stats_path: str, query_idx: int | None) -> None:
    recs = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    recs = [r for r in recs if r.get("state") and r.get("chunk")]
    if not recs:
        print(f"{path}: 没有带 state/chunk 的记录")
        return

    stats_all = json.loads(Path(stats_path).read_text())["norm_stats"]
    s_stats, a_stats = stats_all["state"], stats_all["actions"]

    if query_idx is not None:
        recs = [r for r in recs if r.get("query") == query_idx]
        if not recs:
            print(f"{path}: 没有 query=={query_idx} 的记录")
            return

    r = recs[0]
    state = np.asarray(r["state"], dtype=np.float64)
    chunk = np.asarray(r["chunk"], dtype=np.float64)

    print(f"===== {path}")
    print(f"      记录: step={r.get('step')} query={r.get('query')} t={r.get('t')}s "
          f"exec_mode={r.get('exec_mode')} action_repeat={r.get('action_repeat')} "
          f"freeze_left={r.get('freeze_left')}")
    print(f"      共 {len(recs)} 条可用记录；下面只展开第 1 条\n")

    # ---- 步骤 0：转储里就有的 —— 客户端发出的 state ----
    print("【步骤 0】转储里**有**记录：客户端发出的 state（原始物理量）")
    print(f"   shape={state.shape}   ← 仿真侧 query_site_pos_and_quat 量到的位姿，不是指令")
    print()

    # ---- 步骤 1：Normalize（转储里没有，重建） ----
    if len(state) != len(s_stats["mean"]):
        print(f"   ** state 是 {len(state)} 维，但 norm_stats 是 {len(s_stats['mean'])} 维 —— "
              f"这轮结果不能用来判断策略好坏（见 README 二·五）")
    state_norm = _normalize(state, s_stats)
    print("【步骤 1】转储里**没有**：Normalize 之后的 state（模型真正看到的数）")
    print("   公式: (x - q01)/(q99 - q01 + 1e-6) * 2 - 1        [transforms.py _normalize_quantile]")
    for i, nm in enumerate(ACTION_LAYOUT[:len(state)] + ([STATE_EXTRA] if len(state) > 18 else [])):
        span = float(np.asarray(s_stats["q99"])[i] - np.asarray(s_stats["q01"])[i])
        print(_fmt_row(nm, state[i], state_norm[i], span))
    print()

    # ---- 步骤 2+3：state 离散成文字 + 拼成句子 ----
    disc, state_str = _discretize_state(state_norm)
    prompt = str(r.get("prompt", ""))
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    print("【步骤 2】转储里**没有**：state 被离散成整数（256 个格子，每格 2/256）")
    print("   公式: digitize(state_norm, linspace(-1,1,257)[:-1]) - 1     [tokenizer.py]")
    print(f"   {state_str}")
    print()
    print("【步骤 3】转储里**没有**：最终喂给模型的那条句子")
    print("   （pi0.5 的 state 不是张量，是这句话里的文字 —— 少一维 = 模型看到一条没见过的串）")
    print(f"   {full_prompt!r}")
    print()

    # ---- 步骤 4：模型输出的 chunk（转储里有），与反推的原始输出 ----
    print(f"【步骤 4】转储里**有**记录：模型返回的 chunk   shape={chunk.shape}"
          f"（= 20 步 × 18 维，已 Unnormalize 回米/四元数）")
    print(f"   右边是反推的**归一化空间原始输出**（Normalize 的逆作用在 chunk 上）")
    a_mean = np.asarray(a_stats["mean"], dtype=np.float64)
    a_q01 = np.asarray(a_stats["q01"], dtype=np.float64)
    a_q99 = np.asarray(a_stats["q99"], dtype=np.float64)
    chunk_norm = _denormalize(chunk, a_stats)
    print(f"   {'维度':16s} {'下发给仿真的值(第0步)':>26s} {'原始输出(第0步)':>18s} {'chunk20步跨度':>14s}")
    for i, nm in enumerate(ACTION_LAYOUT):
        span = float(chunk[:, i].max() - chunk[:, i].min())
        print(f"   {nm:16s} {chunk[0, i]:+26.5f} {chunk_norm[0, i]:+18.4f} {span:14.5f}")
    print()
    print(f"   注：18 维之后的填充维（模型输出 32 维）在 "
          f"G1OmnipickerOutputs 里被 [..., :18] 截掉了，转储没留，**无法恢复**。")
    print()
    print("【重建不出来】转储里**永远没有**这两样：")
    print("   * 图像   —— 转储不存 obs 里的相机帧（ResizeImages 到 224×224 之后的样子）")
    print("   * 去噪轨迹 —— flow-matching 的 50 步逐步噪声预测，只在服务端内存里存在过")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--ckpt", default=DEFAULT_STATS, help="norm_stats.json 路径")
    ap.add_argument("--query", type=int, default=None, help="只看第几次推理（默认第 1 条）")
    args = ap.parse_args()
    if not Path(args.ckpt).exists():
        print(f"找不到 norm_stats: {args.ckpt}", file=sys.stderr)
        raise SystemExit(2)
    report(args.dump, args.ckpt, args.query)


if __name__ == "__main__":
    main()
