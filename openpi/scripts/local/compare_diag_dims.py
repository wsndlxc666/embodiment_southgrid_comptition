#!/usr/bin/env python
"""把几个 train_diag*.jsonl 的**逐维 loss**摆在同一张表上比，按 step 对齐。

**这些数是不加权的原始 MSE，可以直接跨档比。** `scripts/train.py:190` 记的是
`jnp.mean(per_dim, axis=(0,1))` —— 未加权的逐维流匹配平方误差；加权只发生在
`jnp.mean(weighted)` 那条 loss 上。所以：
  - 不要拿它去除以 loss_weights（我犯过这个错：给基线除 0.857、给新档除 2.21，
    等于把基线抬高、把新档压低，凭空制造出"新档更好"）；
  - 也别拿它去和日志里的 `loss=` 对账，那个是加权+平均过的。

**必须带对照维**：只盯 dim19/21 会重蹈"窗口开太窄"的覆辙 —— 如果新档在
**所有**维上都更低（比如只是这批样本/热启动更顺），那 index 维变低说明不了任何
和 index 有关的事。所以本脚本默认同时打一组控制维（臂/爪）的均值。

**注意 step 不是"总训练量"**：热启动的档从 19000 接着练，但 step 计数从 0 重开，
所以"新档 step 1000"= 19000+1000 步的模型，和基线自己的 step 1000 不是同一个成熟度。

用法: compare_diag_dims.py [--dims 19,20,21] [--ctl 7,8,16,17] <a.jsonl> <b.jsonl> ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # 训练被中断时最后一行可能是半行
            if "step" in r:
                out[int(r["step"])] = r
    return out


def get(rec: dict, d: int) -> float:
    return rec.get(f"dim{d:02d}", float("nan"))


def mean_of(rec: dict, dims: list[int]) -> float:
    vals = [get(rec, d) for d in dims]
    vals = [v for v in vals if v == v]          # 去掉 nan
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="19,20,21", help="关注维（默认 index 三维）")
    ap.add_argument("--ctl", default="7,8,16,17",
                    help="控制维（默认右臂 x + 右爪；用来判断'是不是整体都变好了'）")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    dims = [int(x) for x in a.dims.split(",") if x.strip()]
    ctl = [int(x) for x in a.ctl.split(",") if x.strip()]

    tags, data = [], {}
    for p in a.files:
        tag = os.path.basename(p).replace("train_diag", "").replace(".jsonl", "").lstrip("_") or "base"
        tags.append(tag)
        data[tag] = load(p)
        if not data[tag]:
            print(f"✗ {p} 一条可用记录都没有", file=sys.stderr)

    common = set.intersection(*(set(data[t]) for t in tags)) if tags else set()
    if not common:
        print("✗ 这几档没有共同的 step，没法逐行对齐")
        return 2
    steps = sorted(common)

    print(f"步数交集 {len(steps)} 个（{steps[0]} … {steps[-1]}）；**原始未加权 MSE，越小越好**\n")
    hdr = f"{'step':>7} |" + "".join(f" {t[:11]:>13} |" for t in tags)
    print(f"{'':>7} |" + "".join(f" {'  '.join(f'd{d}' for d in dims):>13} |" for t in tags))
    print(hdr)
    print("-" * len(hdr))
    for s in steps:
        cells = []
        for t in tags:
            cells.append(f"{'  '.join(f'{get(data[t][s], d):.5f}' for d in dims):>13}")
        print(f"{s:>7} |" + "".join(f" {c} |" for c in cells))

    print(f"\n控制维 {ctl} 的均值（若这里也一起降，那 index 维降就不是 index 的功劳）")
    print(hdr)
    print("-" * len(hdr))
    for s in steps:
        cells = [f"{mean_of(data[t][s], ctl):>13.5f}" for t in tags]
        print(f"{s:>7} |" + "".join(f" {c} |" for c in cells))

    print("\n末步一览")
    last = steps[-1]
    for t in tags:
        print(f"  {t:<14} step={last:<6} " +
              "  ".join(f"d{d}={get(data[t][last], d):.5f}" for d in dims) +
              f"   控制维均值={mean_of(data[t][last], ctl):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
