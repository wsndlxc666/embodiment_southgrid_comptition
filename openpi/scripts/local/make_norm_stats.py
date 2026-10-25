#!/usr/bin/env python
"""直接从 parquet 算 norm_stats，绕开 compute_norm_stats.py 的视频解码。

为什么等价:训练侧 inputs 管线是
    repack_transforms.inputs (只改键名)  ->  G1OmnipickerInputs (state/actions 原样透传)
所以喂给 RunningStats 的就是 parquet 里那两列的原始值,一模一样。
官方脚本慢是因为它走完整 lerobot dataset,每个样本要解 20 步 × 2 路视频。

用法:
    make_norm_stats.py <repo_id> <输出 assets 目录>       # 算并写
    make_norm_stats.py <repo_id> <参考 norm_stats.json>   # 只和参考文件比对
"""
import glob
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize as N


def stats_from_parquet(repo: str):
    st = {k: N.RunningStats() for k in ("state", "actions")}
    files = sorted(glob.glob(f"/home/dgl/lxc/openpi/data/{repo}/data/chunk-*/episode_*.parquet"))
    if not files:
        raise SystemExit(f"没找到 parquet: {repo}")
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "action"])
        st["state"].update(np.array(t["observation.state"].to_pylist(), dtype=np.float64))
        st["actions"].update(np.array(t["action"].to_pylist(), dtype=np.float64))
    return {k: st[k].get_statistics() for k in st}, len(files)


def main() -> None:
    repo, target = sys.argv[1], sys.argv[2]
    mine, n_ep = stats_from_parquet(repo)
    print(f"从 {n_ep} 集 parquet 复算出统计量, state/actions 维度 = {np.asarray(mine['state'].mean).shape}")

    if target.endswith(".json") and "norm_stats" in pathlib.Path(target).name:
        ref = json.load(open(target))["norm_stats"]
        ok = True
        for key in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                a = np.asarray(getattr(mine[key], field))
                b = np.asarray(ref[key][field])
                if a.shape != b.shape:
                    print(f"  {key}.{field}: 形状不符 {a.shape} vs {b.shape}")
                    ok = False
                    continue
                d = float(np.abs(a - b).max())
                flag = "✅" if d < 1e-4 else "❌"
                ok &= d < 1e-4
                print(f"  {key}.{field:<5} 最大偏差 {d:.3e} {flag}")
        print("结论:", "复算方法正确" if ok else "复算方法有问题, 不能用")
    else:
        out = pathlib.Path(target)
        N.save(out, mine)
        print(f"已写入: {out}/norm_stats.json")
        for key in ("state", "actions"):
            q01 = np.asarray(mine[key].q01)
            q99 = np.asarray(mine[key].q99)
            print(f"  {key}: 维度 {len(q01)}")
            print(f"    [16] 右爪内 q01={q01[16]:.6f} q99={q99[16]:.6f}")
            print(f"    [17] 右爪外 q01={q01[17]:.6f} q99={q99[17]:.6f}")
            print(f"    [18] 距离   q01={q01[18]:.6f} q99={q99[18]:.6f}")


if __name__ == "__main__":
    main()
