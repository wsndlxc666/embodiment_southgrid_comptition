#!/usr/bin/env python
"""把 cycle_index.py 的**函数字节码**和 18:08 那份可信 pyc 逐一对拍。

起因：cycle_index.py 第 1 行被写坏 —— 文档字符串开头的三个引号，第一个变成了全角逗号，
于是整行成了 `，""按「周期相位」...`。模块 import 直接 SyntaxError（已从 pyc 证明
18:08 时首行还是干净的）。
注：写这个脚本时我第一版就把上面那个坏行**原样**贴进了自己的文档字符串，里面含三个连续
引号，提前终止了字符串 —— 和 cycle_index.py 坏掉的是同一类事故。故此处不复述原文。
坏的是一个字符，但**没人能保证只坏了那一个字符**，而这个模块正是过采样/反事实增强的
全部实现 —— 带着一个坏掉的增强跑 10 小时训练，比直接报错糟得多。

所以这里不看文档字符串（它本来就被人为改过），只比 `co_code`：只要每个函数的字节码
和 pyc 里的逐字节相同，就说明**代码逻辑没被动过**；顺带比 co_names / co_varnames。

用法：/home/dgl/lxc/openpi/.venv/bin/python scripts/local/check_cycle_index_integrity.py
"""
from __future__ import annotations

import marshal
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src/openpi/training/cycle_index.py"
PYC = ROOT / "src/openpi/training/__pycache__/cycle_index.cpython-311.pyc"


def collect(code: types.CodeType, prefix: str = "") -> dict[str, types.CodeType]:
    out: dict[str, types.CodeType] = {}
    name = f"{prefix}{code.co_name}"
    out[name] = code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out.update(collect(c, f"{name}."))
    return out


def main() -> int:
    if not PYC.exists():
        print(f"✗ 找不到基准 pyc: {PYC}")
        return 2
    with PYC.open("rb") as f:
        f.read(16)  # magic + flags + mtime + size
        ref = collect(marshal.load(f))

    live = collect(compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec"))

    bad = 0
    only_ref = sorted(set(ref) - set(live))
    only_live = sorted(set(live) - set(ref))
    for n in only_ref:
        print(f"✗ 基准里有、现在没有: {n}")
        bad += 1
    for n in only_live:
        print(f"+ 现在多出来的（预期：无）: {n}")

    common = sorted(set(ref) & set(live))
    diff = []
    for n in common:
        a, b = ref[n], live[n]
        if a.co_code != b.co_code:
            diff.append(n)
        elif a.co_names != b.co_names or a.co_varnames != b.co_varnames:
            diff.append(f"{n} (仅名字表不同)")
    if diff:
        print("✗ 字节码不一致的函数:")
        for n in diff:
            print(f"    {n}")
        bad += len(diff)

    print(f"\n比对了 {len(common)} 个代码对象；不一致 {len(diff)} 个；"
          f"基准独有 {len(only_ref)} 个；新增 {len(only_live)} 个")
    if bad == 0:
        print("✓ 代码逻辑与 18:08 那份逐字节一致 —— 这次事故只坏了文档字符串的首字符")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
