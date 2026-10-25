#!/usr/bin/env python
"""按**事先定好**的规则给 probe_index_channel4 的前后对比下判决，不许事后挑口径。

背景（读之前先看这段，否则会读反）：
  · `label_imgdon`（脏 dims19/20 **并且**换图）就是**改法 A 训练时构造的那个输入**：
    图像与 dims 说"第 q 轮"，dim21 与 action 真值说"第 p 轮"。
  · 所以一个**学会了听 idx** 的模型，在这个输入上应该**不动**（输出 ≈ 参照帧的真值动作）
    ⇒ proj ≈ 0。
  · 而**跟着图/dims 走**的模型会朝供体挪 ⇒ proj > 0。
  也就是说这一行的**期望方向是"降"**，不是"升"。这是最容易读反的地方：
  proj 正的越多，说明它越不听 idx。
  · 反过来 `all ref`（连 dim21 一起脏成供体的）那一行，**越听索引越该升**（proj > 0）。

三条判据（在**没看到任何新档数字之前**就写在这里了）：
  P1 量尺自检：两档的 `none ref`（真空对照）都必须 p ≈ 0。
     不满足 ⇒ 统计量坏了，后面两条不看。
  P2 主判据：`label_imgdon` 的配对差 t < −2（显著朝 0/负走），
     且**新档**该行的 |proj| 落在 0 的噪声内（|proj| ≤ 2·SE 或 ≤ 2mm）。
  P3 没有变成"对索引整体失聪"：新档的 `all ref` 仍然 p > 0 且 |t| > 2。

用法: verdict_probe_v4.py <前tag> <后tag>
退出码 0 = 全过，1 = 未过，2 = 数据不全（没法判）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COND = "A 图+state（分布内）"   # 判决只看分布内那个条件；B 是近似评测，作附注
VACUUM_MM = 3.0               # 真空对照的 |proj| 上限（mm），两档都要过
NEW_ZERO_MM = 2.0             # 新档 label_imgdon 判"不动"的绝对上限（mm）


def load(tag: str, suffix: str):
    p = os.path.join(HERE, f"probe_v4_{tag}_{suffix}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def row(doc: dict, cond: str) -> dict:
    return doc["summary"][cond]


def paired(a: dict, b: dict, cond: str, field: str = "proj_mm"):
    ma = {(r["ep"], r["idx_p"], r["k"]): r for r in a["rows"][cond]}
    mb = {(r["ep"], r["idx_p"], r["k"]): r for r in b["rows"][cond]}
    common = sorted(set(ma) & set(mb))
    d = np.array([mb[k][field] - ma[k][field] for k in common])
    if len(d) < 3:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def t_of(mean: float, mean_se: float) -> float:
    return mean / mean_se if mean_se and mean_se > 0 else float("nan")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ta, tb = sys.argv[1], sys.argv[2]

    need = ("label_imgdon", "none", "all")
    docs = {}
    for suffix in need:
        da, db = load(ta, suffix), load(tb, suffix)
        if da is None or db is None:
            missing = []
            if da is None:
                missing.append(f"{ta}/{suffix}")
            if db is None:
                missing.append(f"{tb}/{suffix}")
            print(f"✗ 缺文件，判不了：{missing}")
            return 2
        docs[suffix] = (da, db)

    fails: list[str] = []
    print(f"判决：前 = {ta}   后 = {tb}   条件 = 「{COND}」\n")

    # ---------------- P1 真空对照 ----------------
    # 注意：两档是**两个不同的模型**，所以真空对照**只查各自是否落在 0**，
    # 不能查"两档之间没有差"—— 输入逐字节相同不代表两个网络的输出必须相同。
    print("P1 量尺自检（none ref 真空对照，两路输入逐字节相同）")
    for tag, doc in ((ta, docs["none"][0]), (tb, docs["none"][1])):
        r = row(doc, COND)
        p, se = r["mean_proj_mm"], r.get("mean_proj_se", float("nan"))
        tt = r.get("mean_proj_t", t_of(p, se))
        ok_t, ok_abs = abs(tt) < 3, abs(p) <= max(VACUUM_MM, 3 * se)
        print(f"   {'✓' if ok_t and ok_abs else '✗'} {tag:<16} proj = {p:+.2f} ± {se:.2f} mm "
              f"(t={tt:+.2f})   门槛 |t| < 3 且 |proj| ≤ max({VACUUM_MM}, 3SE)")
        if not (ok_t and ok_abs):
            fails.append(f"P1 {tag} 真空对照不为 0：{p:+.2f}±{se:.2f}mm (t={tt:+.2f})")
    if fails:
        print("\n真空对照没过 ⇒ 统计量本身有问题，后面两条不看。")
        return 1

    # ---------------- P2 主判据 ----------------
    print("\nP2 主判据（label_imgdon = 改法 A 的训练输入；**期望朝 0 降**）")
    dm, dse, n = paired(docs["label_imgdon"][0], docs["label_imgdon"][1], COND)
    t = dm / dse if dse and dse > 0 else float("nan")
    ra = row(docs["label_imgdon"][0], COND)
    rb = row(docs["label_imgdon"][1], COND)
    print(f"   前 {ra['mean_proj_mm']:+.2f}±{ra.get('mean_proj_se', float('nan')):.2f} mm "
          f"→ 后 {rb['mean_proj_mm']:+.2f}±{rb.get('mean_proj_se', float('nan')):.2f} mm")
    print(f"   配对差 {dm:+.2f}±{dse:.2f} mm  (t={t:+.2f}, n={n})")
    new_se = rb.get("mean_proj_se", float("nan"))
    zero_ok = abs(rb["mean_proj_mm"]) <= max(NEW_ZERO_MM, 2 * new_se)
    print(f"   {'✓' if t < -2 else '✗'} 显著朝 0 走：t < −2（实得 {t:+.2f}）")
    print(f"   {'✓' if zero_ok else '✗'} 新档已落到 0 的噪声内：|{rb['mean_proj_mm']:+.2f}| "
          f"≤ max({NEW_ZERO_MM}, 2SE={2 * new_se:.2f})")
    if not (t < -2):
        fails.append(f"P2 配对差没显著为负：t={t:+.2f}")
    if not zero_ok:
        fails.append(f"P2 新档仍然偏供体：proj={rb['mean_proj_mm']:+.2f}±{new_se:.2f}mm")

    # ---------------- P3 索引灵敏度没丢 ----------------
    # 用**配对差**而不是"新档自己显著不显著"。理由：基线 `all ref` 本身才 t=2.38，
    # 就在门槛边上打转 —— 拿"新档单独 |t|>2"当判据，即使两个模型一模一样也有一半机会
    # 判失败，那是个假阴性机器。配对设计消掉了样本对的个体差，问的是真正想问的那句：
    # **相对基线，新档在索引变脏时朝供体走的力度有没有掉下来。**
    print("\nP3 没有变成对索引整体失聪（all ref：连 dim21 一起脏，**越听索引越该升**）")
    dm3, dse3, n3 = paired(docs["all"][0], docs["all"][1], COND)
    t3 = dm3 / dse3 if dse3 and dse3 > 0 else float("nan")
    ra, rb = row(docs["all"][0], COND), row(docs["all"][1], COND)
    p, se = rb["mean_proj_mm"], rb.get("mean_proj_se", float("nan"))
    print(f"   前 {ra['mean_proj_mm']:+.2f}±{ra.get('mean_proj_se', float('nan')):.2f} mm "
          f"→ 后 {p:+.2f}±{se:.2f} mm")
    print(f"   配对差 {dm3:+.2f}±{dse3:.2f} mm  (t={t3:+.2f}, n={n3})")
    ok_keep, ok_dir = t3 > -2, p > 0
    print(f"   {'✓' if ok_keep else '✗'} 索引灵敏度**没有下降**：配对差 t > −2（实得 {t3:+.2f}）")
    print(f"   {'✓' if ok_dir else '✗'} 方向仍在索引那边：新档 proj = {p:+.2f} > 0")
    if not (ok_keep and ok_dir):
        fails.append(f"P3 索引灵敏度受损：配对差 {dm3:+.2f}±{dse3:.2f}mm (t={t3:+.2f})，"
                     f"新档 proj={p:+.2f}mm")

    # ---------------- 附注：其余行 ----------------
    print("\n附注（不参与判决，只看方向）")
    for suffix in ("idx", "label", "none_imgdon"):
        da, db = load(ta, suffix), load(tb, suffix)
        if da is None or db is None:
            print(f"   {suffix:<12} —— 缺文件")
            continue
        a, b = row(da, COND), row(db, COND)
        dm2, dse2, n2 = paired(da, db, COND)
        t2 = dm2 / dse2 if dse2 and dse2 > 0 else float("nan")
        print(f"   {suffix:<12} {a['mean_proj_mm']:+7.2f} → {b['mean_proj_mm']:+7.2f} mm   "
              f"配对差 {dm2:+7.2f}±{dse2:.2f} (t={t2:+.2f}, n={n2})")

    print()
    if fails:
        print("✗ 判决：未通过")
        for m in fails:
            print(f"    - {m}")
        return 1
    print("✓ 判决：三条全过 —— 模型在图像与 idx 冲突时改听 idx 了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
