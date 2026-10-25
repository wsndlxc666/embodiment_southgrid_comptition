#!/usr/bin/env python
"""对比 d19 v1(5个工具参照点) 与 v2(目标集) 两次训练的逐维 loss。

为什么需要专门的对比：两个 run 都从同一个 18 维 checkpoint(clear5_v1_l3090_lora/43500)
热启动，batch/config 全同，**唯一变量就是第 19 维的定义**。所以逐维曲线的差异可以
归因到这个改动上。

注意两条容易踩的坑：
  1. 必须**同 step 对齐**比。训练最前面 ~150 步是瞬态(loss 先升后降)，拿任意两段
     算"降幅"会得出相反的结论。
  2. 单步的值抖动很大(同一维在相邻十步能差 5 倍)，必须用**段均值**，不能看单点。

判据(起训前就定死，避免事后找解释)：
  右爪两维 dim16/dim17 是真正学不动的那两维(v1 跑 13600 步只从 0.080 降到 0.055)。
  v2 在 step 1800~3300 的段均值若**明显低于** v1 同段的 0.0806 -> 改动有效。
  若仍在 0.080 附近 -> 这个输入表征的改动没有转化成行为改善。

用法:  compare_d19_runs.py [v2起始step, 默认 0]
"""
import json
import sys

V1 = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d19_100ep/train_diag.jsonl"
V2 = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d19_goal_100ep/train_diag.jsonl"
DIMS = [("dim16", "右爪内"), ("dim17", "右爪外"), ("dim18", "距离"),
        ("dim07", "右臂x"), ("dim08", "右臂y"), ("dim09", "右臂z")]
WINDOW = 1500


def load(p):
    return {r["step"]: r for r in (json.loads(l) for l in open(p))}


def seg_mean(recs, key, lo, hi):
    vals = [r[key] for s, r in recs.items() if lo <= s < hi]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    a, b = load(V1), load(V2)
    print(f"v1(5工具) step {min(a)}~{max(a)}   {len(a)} 条记录")
    print(f"v2(目标集) step {min(b)}~{max(b)}   {len(b)} 条记录\n")

    top = max(b)
    print(f"=== 段均值对比 (每段 {WINDOW} 步) ===")
    # 只比**两边都跑满**的段。否则会拿 v1 的稳态去比 v2 的瞬态，
    # 得出完全相反的结论(最前面的瞬态 loss 本来就高)。
    full = [lo for lo in range(0, top + 1, WINDOW) if lo + WINDOW - 1 <= top]
    if not full:
        print(f"  v2 目前只到 step {top}，还不够一个完整的 {WINDOW} 步段。")
        print(f"  跑满第一段预计还需 ~{(WINDOW - top) * 1.6 / 60:.0f} 分钟。")
    else:
        hdr = "  ".join(f"{n:>16}" for _, n in DIMS)
        print(f"{'step 段':>15}  {hdr}")
        for lo in full:
            hi = lo + WINDOW
            cells = []
            for key, _ in DIMS:
                m1 = seg_mean(a, key, lo, hi)
                m2 = seg_mean(b, key, lo, hi)
                cells.append(f"{m1:.4f}->{m2:.4f}")
            print(f"{lo:>6}-{hi:<6}  " + "  ".join(f"{c:>16}" for c in cells))
        print("  (格式: v1 -> v2, 同一段; 均为跑满的段)")

    # 判据
    print("\n=== 判据: 右爪在 step 1800~3300 ===")
    LO, HI = 1800, 3300
    if not any(LO <= s < HI for s in b):
        print(f"  v2 还没跑到 step {LO} (当前 {top})，无法判定。")
    else:
        for key, name in (("dim16", "右爪内"), ("dim17", "右爪外")):
            m1, m2 = seg_mean(a, key, LO, HI), seg_mean(b, key, LO, HI)
            rel = (m2 - m1) / m1 * 100
            verdict = "✅ v2 明显更低" if rel < -10 else ("❌ 没有改善" if rel > -3 else "⚠️ 略有改善但不足以定论")
            print(f"  {name}: v1={m1:.4f}  v2={m2:.4f}  相对变化 {rel:+.1f}%   {verdict}")

    print("\n提醒: 逐维 loss 只是间接证据。真正要回答的是仿真里右臂有没有"
          "\n从左到右依次抓取并闭爪 —— 那要靠评测, 不能只看曲线。")


if __name__ == "__main__":
    main()
