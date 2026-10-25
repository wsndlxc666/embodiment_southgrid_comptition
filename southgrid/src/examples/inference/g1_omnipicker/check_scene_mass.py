#!/usr/bin/env python
"""场景 XML 里的质量/惯量：采集时那次 vs 评测那次，是不是同一套。

为什么要查：用户的观察是"策略抓取时手臂一直往下掉，比训练时重"。
如果 OrcaLab 每次重建场景都会重新随机化模型，而随机化**连质量一起抽**，
那训练（100 集在**同一次会话**里采完）与评测（另一次抽样）之间就存在系统性
动力学差 —— 重力补偿对不上，手臂就会在负载下塌。

本脚本只读 ~/.orcagym/tmp/*.xml，不起仿真、不碰机器人。

⚠ 工具刚体**没有 `<inertial mass=...>`**：它们的质量来自 `<geom density="1000">`
  乘 geom 体积。所以必须按 density×体积 算，只看 `<inertial>` 会全部读到"—"
  （v1 就是这么错的）。另外解析属性时要用 `(?<![\\w])type=` —— 否则
  `contype="1"` 里的 "type" 会被当成 geom 的 type，体积公式全用错。

判据：
  · 各次 XML 里同一件工具的**质量完全相同** ⇒ 质量不是被随机的那一项。
  · 质量**每次不同** ⇒ 动力学不是同一套。

2026-09-21 实测结论（225 份 XML，08-26 ~ 09-21，其中 216 份含机器人）：
  · 工具质量**只有 1 种取值**：扳手 0.265694 / 螺丝刀 0.532373 / 手电筒 0.123784 /
    电工刀01 0.189897 / 电工刀02 0.170929 kg。
  · 右臂 6 节连杆的 `<inertial>` 也**逐字节相同**（link1 mass=0.49360001 …）。
  ⇒ **质量/惯量没有被随机化，训练与评测是同一套。**
  手臂在评测里"显得更重"的真因见 measure_osc_droop.py 与本仓 runner:750 的 --kp。

用法：
  /home/dgl/miniconda3/envs/orcalab/bin/python check_scene_mass.py
  ... --grep arm_r_link        # 也看机械臂连杆
  ... --all                    # 扫全部 XML（默认只扫最近 8 份）
"""
from __future__ import annotations

import argparse
import datetime
import glob
import math
import os
import re
import sys

TMP = os.path.expanduser("~/.orcagym/tmp")

# 工具体名 -> 中文
TOOLS = {
    "Spanner": "扳手",
    "Screwdriver": "螺丝刀",
    "Flashlight": "手电筒",
    "ElectriciansKnife01": "电工刀(左)",
    "ElectriciansKnife02": "电工刀(右)",
}

VOL = {
    "box": lambda v: 8 * v[0] * v[1] * v[2],
    "cylinder": lambda v: math.pi * v[0] ** 2 * 2 * v[1],
    "sphere": lambda v: 4 / 3 * math.pi * v[0] ** 3,
    "capsule": lambda v: math.pi * v[0] ** 2 * 2 * v[1] + 4 / 3 * math.pi * v[0] ** 3,
    "ellipsoid": lambda v: 4 / 3 * math.pi * v[0] * v[1] * v[2],
    "mesh": lambda v: 0.0,
}


def attr(tag: str, key: str):
    """取属性。`(?<![\\w])` 是必需的 —— 否则 contype= 里的 type 会被匹配到。"""
    m = re.search(r"(?<![\w])" + key + r'="([^"]*)"', tag)
    return m.group(1) if m else None


def tool_mass(xml: str, key: str):
    """工具 body 下所有 geom 的 density×体积 之和 (kg)。"""
    m = re.search(r'<body\b[^>]*?name="[^"]*%s[^"]*_joint"' % key, xml)
    if not m:
        return None
    s = m.start()
    nxt = xml.find("<body", m.end())
    end = xml.find("</worldbody>", s)
    e = min([x for x in (nxt, end, len(xml)) if x and x > 0])
    tot = 0.0
    for g in re.findall(r"<geom\b([^>]*?)/>", xml[s:e], re.S):
        ty = attr(g, "type") or "sphere"
        d, sz = attr(g, "density"), attr(g, "size")
        if d is None or sz is None:
            continue
        f = VOL.get(ty)
        if f:
            tot += float(d) * f([float(x) for x in sz.split()])
    return tot


def bodies(xml: str):
    """(name, inertial_tag) 列表 —— 每个 body 只取**它自己**的第一个 inertial。"""
    out = []
    pos = [(m.start(), m.group(1)) for m in re.finditer(r'<body\b[^>]*?name="([^"]+)"', xml)]
    for i, (s, name) in enumerate(pos):
        e = pos[i + 1][0] if i + 1 < len(pos) else len(xml)
        mi = re.search(r"<inertial\b[^>]*?/>", xml[s:e])
        out.append((name, mi.group(0) if mi else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", type=str, default="", help="额外名字过滤（如 arm_r_link）")
    ap.add_argument("--n", type=int, default=8, help="看最近几个 XML")
    ap.add_argument("--all", action="store_true", help="扫全部 XML 并按取值分组汇总")
    args = ap.parse_args()

    allf = sorted(glob.glob(os.path.join(TMP, "*.xml")), key=os.path.getmtime)
    files = allf if args.all else allf[-args.n:]
    if not files:
        print(f"✗ {TMP} 下没有 xml")
        sys.exit(1)

    checked = []
    table = {}
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            xml = fh.read()
        if 'name="g1_omnipicker_arm_r_link1"' not in xml:
            continue                      # 不含机器人的场景，跳过
        checked.append(f)
        row = {}
        for key, cn in TOOLS.items():
            row[cn] = tool_mass(xml, key)
        table[f] = (os.path.getsize(f), os.path.getmtime(f), row)

    if not checked:
        print("✗ 这几份 XML 里都没有机器人（无 arm_r_link1），换个 --n 或加 --all")
        sys.exit(1)

    ts = lambda f: datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%m-%d %H:%M")
    span = f"{ts(checked[0])} ~ {ts(checked[-1])}"
    print(f"检查 {len(checked)} 份含机器人的场景 XML（{span}）\n")

    if args.all:
        # 按"质量取值组合"分组 —— 一眼看出有没有被随机化
        groups = {}
        for f in checked:
            sig = tuple(None if v is None else round(v, 6) for v in table[f][2].values())
            groups.setdefault(sig, []).append(f)
        print(f"=== 工具质量的**不同取值**种数：{len(groups)} ===")
        for sig, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  出现 {len(fs):>3} 次 ({ts(fs[0])} ~ {ts(fs[-1])}): "
                  + ", ".join(f"{n}={v:.6f}" for n, v in zip(TOOLS.values(), sig) if v is not None))
        if len(groups) == 1:
            print("\n✓ 质量**每次完全相同** —— 没有被随机化，训练与评测是同一套动力学。")
        else:
            print(f"\n⚠ 有 {len(groups)} 种不同取值 —— 质量在被随机化！")
    else:
        names = list(TOOLS.values())
        print(f"{'时间':<12}{'字节':>8}  " + "  ".join(f"{n:>12}" for n in names))
        for f in checked:
            sz, mt, row = table[f]
            print(f"{ts(f):<12}{sz:>8}  " + "  ".join(
                f"{row[n]:>12.6f}" if row.get(n) is not None else f"{'—':>12}" for n in names))
        ref = table[checked[-1]][2]
        bad = sum(1 for f in checked for n in names
                  if ref.get(n) is not None and table[f][2].get(n) is not None
                  and abs(ref[n] - table[f][2][n]) > 1e-9)
        print()
        print(f"✓ 逐位相同（{bad} 处不同）" if not bad else f"⚠ {bad} 处质量不同")
        print("  --all 可以扫全部 XML 并按取值分组，结论更硬。")

    if args.grep:
        print(f"\n=== 名字含 '{args.grep}' 的 body 的 inertial ===")
        groups = {}
        for f in checked:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                hit = tuple((n, t) for n, t in bodies(fh.read()) if args.grep in n)
            groups.setdefault(hit, []).append(f)
        print(f"  不同取值种数：{len(groups)}")
        for hit, fs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]:
            print(f"  出现 {len(fs):>3} 次 ({ts(fs[0])} ~ {ts(fs[-1])}):")
            for n, t in hit[:3]:
                m = re.search(r'mass="([^"]+)"', t or "")
                print(f"      {n:<40} mass={m.group(1) if m else '—'}")


if __name__ == "__main__":
    main()
