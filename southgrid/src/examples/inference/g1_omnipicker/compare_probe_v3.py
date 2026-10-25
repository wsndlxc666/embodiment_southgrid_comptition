#!/usr/bin/env python
"""对比两次 probe_index_channel3 的结果（训练前 vs 训练后）。

**用配对比较**：两次跑的是同一份计划（同样的集/相位/k ⇒ 同样的参照帧和供体帧），
所以逐对相减能消掉"这一对本来就好量/难量"的差异，比拿两个独立的均值比有力得多。

判读（见 probe_index_channel3.py 文件头的 SWAP 说明）：

  训练时的干预只弄脏 dims19/20，dim21(cycle_idx) 始终是真值。所以期望的训练后签名是
      idx   斜率↑（模型更听 cycle_idx）
      label 斜率↓（模型不再那么听 goal_y）
      all   斜率↑（索引块整体更有权柄）

  如果 label 不降反升、idx 不动 ⇒ 学到的还是 goal_y 那条捷径，干预没打中。

用法：
  python compare_probe_v3.py base_d22_19000 d22idx3k
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ("A 图+state（分布内）", "B 图置零（近似评测）")


def load(tag: str, suffix: str):
    p = os.path.join(HERE, f"probe_v3_{tag}_{suffix}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def key(r):
    return (r["ep"], r["idx_p"], r["k"])


def paired(a_rows, b_rows, field="slope"):
    """逐对相减（b - a），按 (集,相位,k) 对齐。返回 (均值, 对间SE, n)。"""
    ma = {key(r): r for r in a_rows}
    mb = {key(r): r for r in b_rows}
    common = sorted(set(ma) & set(mb))
    d = np.array([mb[k][field] - ma[k][field] for k in common])
    if len(d) < 3:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    tag_a, tag_b = sys.argv[1], sys.argv[2]
    print(f"对比：A = {tag_a}（前）   B = {tag_b}（后）\n")
    print(f"{'换什么':<16} {'条件':<8} {'前 斜率':>13} {'后 斜率':>13} "
          f"{'配对差':>13} {'t':>6} {'前→后 朝供体':>13}")
    for suffix, label in (("all", "索引块(18:22)"),
                          ("idx", "只 cycle_idx(21)"),
                          ("label", "只 goal_y(19,20)"),
                          ("none_imgdon", "只图像(索引不动)")):
        da, db = load(tag_a, suffix), load(tag_b, suffix)
        if da is None or db is None:
            print(f"{label:<16} —— 缺文件（{tag_a if da is None else tag_b} / {suffix}）")
            continue
        for cond in CONDS:
            ra, rb = da["rows"][cond], db["rows"][cond]
            sa, sb = da["summary"][cond], db["summary"][cond]
            dm, dse, n = paired(ra, rb)
            t = dm / dse if dse and dse > 0 else float("nan")
            mark = ""
            if abs(t) > 2:
                mark = "  ← 显著" + ("↑" if dm > 0 else "↓")
            print(f"{label:<16} {cond[:4]:<8} "
                  f"{sa['mean_slope']:>5.3f}±{sa['mean_slope_se']:.3f} "
                  f"{sb['mean_slope']:>5.3f}±{sb['mean_slope_se']:.3f} "
                  f"{dm:>+7.3f}±{dse:.3f} {t:>6.2f} "
                  f"{100*sa['toward_rate']:>4.0f}%→{100*sb['toward_rate']:<4.0f}%{mark}")
    print("""
（配对差 = 后 − 前，同一对样本相减；t>2 才算量到了变化）

判读：
  · 「只 cycle_idx(21)」↑ 而「只 goal_y(19,20)」↓ ⇒ 模型学会了听 cycle_idx ← 目标
  · 「只 goal_y」不降反升                  ⇒ 还是走 goal_y 捷径，干预没打中
  · 「只图像」也涨了                        ⇒ 模型改用视觉解的，不是索引
  对照基准：训练时被弄脏的只有 19/20，21 一直是真值；所以「只 goal_y」这一行
  正是训练构造的那个条件，它该是降得最明显的一条。""")


if __name__ == "__main__":
    main()
