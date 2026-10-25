#!/usr/bin/env python3
"""把 parquet 的 schema metadata 抹成 None，与源数据集对齐。

背景：`build_phase_shuffled_dataset.py` 里 `_strip_meta()` 是**构建跑到一半才加上**的，
所以早于那一刻写出的分片仍带 pandas 的 `pandas` metadata 块。列名/列型/行数完全一致，
多余 metadata 一般无害，但源是 `metadata = None`，保持一致免得日后排查时多一个变量。

幂等：已经是 None 的文件直接跳过（不重写，省时间也避免动 mtime）。

用法:
    strip_parquet_meta.py [目录...]        # 默认 idxS 数据集的 data/chunk-000
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT = [Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep/data/chunk-000")]


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT
    files = sorted(p for d in dirs for p in d.glob("episode_*.parquet"))
    if not files:
        print(f"✗ 没找到 parquet：{dirs}")
        return 1

    changed, already, failed = 0, 0, []
    for p in files:
        try:
            if pq.read_schema(p).metadata is None:
                already += 1
                continue
            t = pq.read_table(p)
            pq.write_table(t.replace_schema_metadata(None), p)
            changed += 1
        except Exception as e:  # noqa: BLE001
            failed.append((p.name, repr(e)))

    print(f"共 {len(files)} 个：新抹掉 {changed}，本来就是 None {already}，失败 {len(failed)}")
    for name, err in failed:
        print(f"  ✗ {name}: {err}")

    # 复核
    bad = [p.name for p in files if pq.read_schema(p).metadata is not None]
    print("✓ 全部为 None" if not bad else f"✗ 仍有 metadata 的: {bad[:10]}")
    return 0 if (not bad and not failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
