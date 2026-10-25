#!/usr/bin/env python
"""读 clear_eval 的 --dump_chunks，按**终局工具坐标**复核"入箱 N/5"到几把。

为什么需要它：`[已入箱]` 那个计数会**偶发误报**（2026-09-14 见过一轮报 3/5 真值 1/5，
详见 README 二·七）。唯一靠得住的复核办法是看转储最后一条里 5 把工具相对底座的位置，
对照标称槽位和箱子 —— 这件事手工做很容易偷懒，所以固化成脚本。

用法：
    python check_final_positions.py <dump.jsonl> [<dump2.jsonl> ...]

判定口径（base 系；base = 底座本体，不是世界系）：
    在箱里 : xy 落进工具箱检测区 x[0.390,0.899] y[0.057,0.469]
    没动   : 距自己的标称槽位 < 5 cm
    别处   : 其余（含被甩出场景 —— 会同时打出距箱心的距离，一眼能看出是几米还是几十厘米）

注意"没动"是**否定性**证据：工具没动却出现在 `[已入箱]` 里，那一定是误报。
反过来"距槽位 3 cm"也可能是被抓起来又放回去了，得连 `[已入箱]` 的 step 一起看。
"""

import json
import sys

import numpy as np

# 与 clear_eval / 采集器同一套标称槽位（TOOL_REF_POS_B 的 xy），顺序 = 转储里的 tool_names
SLOT_XY = {
    "扳手": (0.564, -0.085),
    "螺丝刀": (0.560, -0.190),
    "电工刀(左)": (0.580, -0.300),
    "手电筒": (0.510, -0.410),
    "电工刀(右)": (0.580, -0.520),
}
# 转储里 `tool_names` 常常是**没有**的（老版本转储只写 tools_world）。那时上面那个
# 按名字查表会全部落空 ⇒ `距槽位=nan`、每把都判成"别处"，"工具根本没动"这条最关键
# 的证据就被吞掉了。工具顺序是**按构造定死**的（见 analyze_chunks.py 的参照点表：
# #0 扳手 … #4 电工刀(右)），所以名字缺了就按位置回退，不许静默变成 nan。
TOOL_ORDER = ("扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)")

# 工具箱检测区（base 系，无余量；clear_eval 启动时打印的就是这几个数）
BOX_XY_MIN = (0.390, 0.057)
BOX_XY_MAX = (0.899, 0.469)
# 工具躺在桌面上的静止高度（base 系 z），用来一眼区分"在地上"和"举在半空"
REST_Z = 0.136


def _canon_name(names: list[str], i: int) -> str | None:
    """第 i 把工具的标准名：名字认得就用名字，认不得就按构造顺序回退。"""
    nm = names[i] if i < len(names) else None
    if nm in SLOT_XY:
        return nm
    return TOOL_ORDER[i] if i < len(TOOL_ORDER) else None


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def report(path: str) -> None:
    recs = _load(path)
    good = [r for r in recs if r.get("tools_world") and r.get("base_world")]
    if not good:
        print(f"== {path}\n   (没有可用记录：{len(recs)} 条里没有 tools_world/base_world)")
        return

    r = good[-1]
    names = r.get("tool_names") or [f"tool{i}" for i in range(len(r["tools_world"]))]
    base = np.asarray(r["base_world"], dtype=np.float64)
    print(f"== {path}")
    fl = r.get("freeze_left")
    fl_note = "未记录(旧版转储，分不清冻结与否)" if fl is None else fl
    print(f"   可用记录 {len(good)}/{len(recs)}，末条 step={r.get('step')} "
          f"(t={r.get('t')}s)  freeze_left={fl_note}")
    if len(good[-1].get("state") or []) < 19:
        print("   ** state 不是 19 维，这轮结果不能用来判断策略好坏（见 README 二·五）")

    n_in_box = 0
    for i, pos in enumerate(r["tools_world"]):
        raw_name = names[i] if i < len(names) else f"tool{i}"
        canon = _canon_name(names, i)
        # 打标准名：转储里经常只有 tool0..tool4，光看名字认不出是哪把工具
        name = canon or raw_name
        if pos is None:
            print(f"   {name:12s} 位置不可用")
            continue
        d = np.asarray(pos, dtype=np.float64) - base
        slot = SLOT_XY.get(canon) if canon else None
        off = float(np.hypot(d[0] - slot[0], d[1] - slot[1])) if slot else float("nan")
        in_box = (BOX_XY_MIN[0] <= d[0] <= BOX_XY_MAX[0]
                  and BOX_XY_MIN[1] <= d[1] <= BOX_XY_MAX[1])
        if in_box:
            verdict = "在箱里 ✓"
            n_in_box += 1
        elif off < 0.05:
            verdict = "没动"
        elif d[2] < REST_Z - 0.05:
            verdict = "掉到桌面下 ✗"
        else:
            verdict = "别处"
        print(f"   {name:12s} base系=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})"
              f"  距槽位={off * 100:6.1f}cm  判定: {verdict}")
    print(f"   → 终局在箱里 {n_in_box}/{len(r['tools_world'])}"
          f"（以此为准，不以上面的 [已入箱] 计数为准）")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for path in sys.argv[1:]:
        report(path)
        print()


if __name__ == "__main__":
    main()
