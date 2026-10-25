"""直读 parquet 算 norm_stats，绕开官方脚本的整条 dataset 管线（含 HEVC 视频解码）。

**为什么不能自己写 np.quantile**：`openpi.shared.normalize.RunningStats` 的分位数
不是真分位数，是 [min,max] 上 5000 个 bin 的直方图取 bin 边（`edges[idx]`），
而且 `_adjust_histograms` 会随 min/max 变化重分布旧计数 —— 结果**依赖喂入顺序**。
所以这里直接 import 同一个类，保证算法逐字一致，只把数据来源换成 parquet。

为什么要这个而不是官方脚本：`scripts/compute_norm_stats.py` 要走完整 dataset 管线，
RepackTransform 里带 "images" 键 ⇒ 每个 batch 都解 2 路 HEVC 视频帧。2026-09-23 实测
**75.75 s/batch，1033 个 batch ⇒ ETA 21 小时 42 分**，不可用。而 state/action 统计
根本不需要图像。

正确性：拿 clear5_train(181 集) 对拍已有的 ground truth，q01/q99 偏差在 1.3e-5 ~ 3e-4
（bin 宽 =(max-min)/5000 的量级）。**关键是 action 必须逐帧原值、不做 action_horizon
分块** —— 见 action_samples() 的注释，那是四个候选规则里唯一对得上的。

用法：
    python norm_stats_direct.py --repo g1_omnipicker_clear5_train \\
        --verify assets/pi05_g1_omnipicker_clear5_lora/g1_omnipicker_clear5_train/norm_stats.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/dgl/lxc/openpi/src")
from openpi.shared import normalize  # noqa: E402

DATA_ROOT = pathlib.Path("/home/dgl/lxc/openpi/data")


def load_episode(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """返回一集的 (state[N,18], action[N,18])。"""
    df = pd.read_parquet(path, columns=["observation.state", "action"])
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ac = np.stack(df["action"].to_numpy()).astype(np.float64)
    return st, ac


def action_samples(ac: np.ndarray) -> np.ndarray:
    """action 统计用**逐帧原值**，不做任何 action_horizon 分块。

    这条是 2026-09-23 在 clear5_train 上对拍 ground truth 反解出来的，四个候选
    规则里只有它成立（判据取右臂 R_x 的 q01，参考值 0.3816）：
        (a) 末端夹最后一帧   0.4408   ✗
        (b) 越界整块丢弃     0.4407   ✗
        (c) 逐帧原值         0.3816   ✓  (Δ1.3e-5，一个 bin 宽)
        (d) 循环回卷         0.3816   ✓  (同上)
    为什么分块会错：分块等于对时间轴做滑动平均，每帧被复制进最多 20 个块，
    平滑把右臂 x 的低尾抹掉了 ⇒ q01 被抬高。旁证：参考文件里 state 与 actions 的
    mean 几乎逐位相同([0.1323, 0.9715, ...])，正因为 action 就是下一帧的 state、
    且两边都没分块。
    """
    return ac


def compute(repo: str) -> dict:
    root = DATA_ROOT / repo
    eps = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not eps:
        raise SystemExit(f"✗ 没找到 parquet：{root}/data/chunk-000/")
    print(f"  读 {len(eps)} 集 …")

    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    n_frames = 0
    for p in eps:
        st, ac = load_episode(p)
        stats["state"].update(st)
        stats["actions"].update(action_samples(ac))
        n_frames += len(st)
    print(f"  共 {n_frames} 帧")
    return {k: s.get_statistics() for k, s in stats.items()}


def as_arrays(ns: dict) -> dict:
    return {k: {kk: np.asarray(getattr(v, kk)) for kk in ("mean", "std", "q01", "q99")}
            for k, v in ns.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", help="输出目录（写 <out>/norm_stats.json）")
    ap.add_argument("--verify", help="已有的 norm_stats.json，对比等价性")
    args = ap.parse_args()

    print(f"=== 直读算 norm_stats: {args.repo} ===")
    got = as_arrays(compute(args.repo))

    if args.verify:
        ref = json.load(open(args.verify))["norm_stats"]
        print(f"\n=== 对拍 {args.verify} ===")
        worst = 0.0
        for key in ("state", "actions"):
            for kk in ("mean", "std", "q01", "q99"):
                a, b = got[key][kk], np.asarray(ref[key][kk])
                if a.shape != b.shape:
                    print(f"  ✗ {key}.{kk} 维度不一致 {a.shape} vs {b.shape}")
                    return 1
                d = np.abs(a - b).max()
                worst = max(worst, d)
                flag = "✓" if d < 1e-6 else ("~" if d < 1e-4 else "✗")
                print(f"  {flag} {key}.{kk:<5} max|Δ| = {d:.3e}")
        # 训练只读 q01/q99，单独报这两个
        q = max(np.abs(got[key]["q" + s] - np.asarray(ref[key]["q" + s])).max()
                for key in ("state", "actions") for s in ("01", "99"))
        print(f"\n  → 训练真正读的 q01/q99 最大偏差 = {q:.3e}")
        print("  → 判定：" + ("等价，可用于新集" if q < 1e-4 else "不等价，别用！"))
        return 0 if q < 1e-4 else 1

    if args.out:
        out = pathlib.Path(args.out)
        normalize.save(out, got)
        print(f"\n✓ 已写 {out}/norm_stats.json")
        for key in ("state", "actions"):
            print(f"  {key}.q01 = {np.round(got[key]['q01'], 4).tolist()}")
            print(f"  {key}.q99 = {np.round(got[key]['q99'], 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
