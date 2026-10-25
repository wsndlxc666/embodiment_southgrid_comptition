#!/usr/bin/env python
"""对比两次 probe_index_channel4 的结果（训练前 vs 训练后），配对比较。

主指标 frac = 输出位移在「供体真值轨迹 − 参照真值轨迹」方向上的**有符号**投影，
除以轨迹差。零假设严格为 0（v3 那个 ‖Δ‖/Δgoal 有 ~0.23 的正地板，不能用）。

判读（训练时只弄脏 dims19/20，dim21 一直是真值）：
  · label ↓ 且 idx ↑        ⇒ 模型学会了在 19/20 与 21 冲突时听 21 ← 目标
  · label 不动或反升         ⇒ 还是走 goal_y 捷径，干预没打中
  · none_don 也涨            ⇒ 模型改用视觉解的，不是索引
  · none_ref 必须 ≈ 0        ⇒ 量尺自检；不 ≈ 0 说明统计量坏了

用法: python compare_probe_v4.py <前tag> <后tag>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = (("all", "索引块(18:22)"), ("idx", "只 cycle_idx(21)"),
        ("label", "只 goal_y(19,20)"), ("none", "真空对照"),
        ("none_imgdon", "只图像(索引不动)"),
        # 文件名的后缀规则见 probe_index_channel4.py:203 —— image=="don" 时加 "_imgdon"。
        # label_imgdon = 脏 19/20 **并且**换图 = **改法 A(d22idxI) 训练时构造的正是这个输入**。
        # 它和 label（= pilot 训练时构造的输入）配成一对，能直接看出换图那一步有没有用。
        ("label_imgdon", "19/20+图(改法A)"))
CONDS = ("A 图+state（分布内）", "B 图置零（近似评测）")


def load(tag, suffix):
    p = os.path.join(HERE, f"probe_v4_{tag}_{suffix}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def paired(a, b, field="proj_mm"):
    ma = {(r["ep"], r["idx_p"], r["k"]): r for r in a}
    mb = {(r["ep"], r["idx_p"], r["k"]): r for r in b}
    common = sorted(set(ma) & set(mb))
    d = np.array([mb[k][field] - ma[k][field] for k in common])
    if len(d) < 3:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    ta, tb = sys.argv[1], sys.argv[2]
    print(f"对比：前 = {ta}   后 = {tb}")
    print("主指标 proj（mm，输出位移在供体真值方向上的投影；零假设 = 0）\n")
    print(f"{'换什么':<18} {'条件':<6} {'前 proj':>16} {'后 proj':>16} "
          f"{'配对差':>15} {'t':>6} {'前→后 朝供体':>13}")
    for suffix, label in ROWS:
        da, db = load(ta, suffix), load(tb, suffix)
        if da is None or db is None:
            print(f"{label:<18} —— 缺文件（{'前' if da is None else '后'} / {suffix}）")
            continue
        for cond in CONDS:
            ra, rb = da["rows"][cond], db["rows"][cond]
            sa, sb = da["summary"][cond], db["summary"][cond]
            dm, dse, n = paired(ra, rb, "proj_mm")
            t = dm / dse if dse and dse > 0 else float("nan")
            mark = "  ←" + ("↑" if dm > 0 else "↓") if abs(t) > 2 else ""
            pa, pb = sa.get("mean_proj_mm", float("nan")), sb.get("mean_proj_mm", float("nan"))
            qa, qb = sa.get("mean_proj_se", float("nan")), sb.get("mean_proj_se", float("nan"))
            print(f"{label:<18} {cond[:4]:<6} "
                  f"{pa:>+8.2f}±{qa:.2f}mm {pb:>+8.2f}±{qb:.2f}mm "
                  f"{dm:>+8.2f}±{dse:.2f}mm {t:>6.2f} "
                  f"{100*sa['toward_rate']:>4.0f}%→{100*sb['toward_rate']:<4.0f}%{mark}")

    print("\n     （附注：同一批数据的 frac 口径，0 = 完全没听索引，1 = 完全跟到供体轨迹）")
    for suffix, label in ROWS:
        da, db = load(ta, suffix), load(tb, suffix)
        if da is None or db is None:
            continue
        for cond in CONDS:
            sa, sb = da["summary"][cond], db["summary"][cond]
            dm, dse, n = paired(da["rows"][cond], db["rows"][cond], "frac")
            t = dm / dse if dse and dse > 0 else float("nan")
            print(f"    {label:<18} {cond[:4]:<6} "
                  f"{sa['mean_frac']:>+7.3f}→{sb['mean_frac']:>+7.3f}  "
                  f"配对差 {dm:>+7.3f}±{dse:.3f} (t={t:+.1f}, n={n})")

    print("""
说明：
  proj 是 mm，零假设严格为 0；「配对差」是同一对样本相减后的均值±SE，|t|>2 才算量到变化。
  「真空对照」那两行是量尺自检：两路输入逐字节相同，前后都该 ≈0，配对差也该 ≈0。
  训练构造的条件就是「只 goal_y(19,20)」那一行 —— 干预若打中，它该降。""")


if __name__ == "__main__":
    main()
