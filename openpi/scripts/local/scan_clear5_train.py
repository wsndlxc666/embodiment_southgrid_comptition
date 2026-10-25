#!/usr/bin/env python3
"""clear5_train 轻量数值粗筛(零 sim, 只读 parquet, ~1-2 分钟)。

只挑"数值/结构上确定坏"的集，不碰几何对错(那需要回放)：
  A. 结构：episode 长度 != 2101、缺列、episode_index 错乱。
  B. 数值：state/action 含 NaN/inf。
  C. 极值：某集 max|action| 相对全集分布成离群(> Q3+5×IQR 的 log 空间)——
     通常是数据损坏/位姿爆掉的特征，真实遥操作动作不会到那个量级。
  D. 冻结：整集某关节动作/状态零方差(全 0)——采集器若让某维一直不动会埋坏学习。

只打印报告、不删文件。判定由调用侧据此做(剔什么)。
"""
import glob, json, math, os, sys
import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
EXPECT = 2101

files = sorted(glob.glob(os.path.join(ROOT, "data", "**", "episode_*.parquet"), recursive=True))
print(f"[scan] {len(files)} 个 episode parquet", flush=True)

rows = []          # (idx, length, nan_count, max_abs_action)
problems = {"A_len": [], "A_col": [], "B_nan": []}

for f in files:
    idx = int(os.path.basename(f).split(".")[0].split("_")[1])
    try:
        t = pq.read_table(f, columns=["observation.state", "action", "episode_index"])
    except Exception as e:
        problems["A_col"].append((idx, str(e))); continue
    if t.column_names != ["observation.state", "action", "episode_index"]:
        problems["A_col"].append((idx, f"缺列 {t.column_names}")); continue
    n = t.num_rows
    state = np.array(t.column("observation.state").to_pylist(), dtype=np.float64)
    act = np.array(t.column("action").to_pylist(), dtype=np.float64)
    if n != EXPECT:
        problems["A_len"].append((idx, n))
    nan = int(np.isnan(state).sum() + np.isnan(act).sum())
    inf = int(np.isinf(state).sum() + np.isinf(act).sum())
    if nan or inf:
        problems["B_nan"].append((idx, f"nan={nan} inf={inf}"))
    rows.append((idx, n, nan, float(np.abs(act).max()) if act.size else 0.0))

print(f"[scan] 正常读取 {len(rows)} 集", flush=True)

# C: log10(max|action|) 稳健离群
logs = np.array([math.log10(max(r[3], 1e-6)) for r in rows])
q1, q3 = np.percentile(logs, 25), np.percentile(logs, 75)
cut = q3 + 5.0 * (q3 - q1)
extreme = [(r[0], r[3]) for r, lg in zip(rows, logs) if lg > cut]

# D: 整集零方差的维(冻结)
frozen = []
for f in files:
    idx = int(os.path.basename(f).split(".")[0].split("_")[1])
    t = pq.read_table(f, columns=["action"])
    act = np.array(t.column("action").to_pylist(), dtype=np.float64)
    zvar = np.where(np.nanstd(act, axis=0) == 0)[0]
    if len(zvar) >= 6:  # 18 维中 ≥6 维整集不动 → 可疑
        frozen.append((idx, len(zvar)))

print("\n========== 粗筛报告 ==========")
for k, v in problems.items():
    print(f"[{k}] {len(v)} 集")
    for it in v[:30]: print("   ", it)
print(f"[C_极值动作离群] {len(extreme)} 集  (cut≈10^{cut:.2f})")
for it in extreme[:30]: print("    idx", it[0], "max|action|=", it[1])
print(f"[D_冻结] {len(frozen)} 集 (≥6维整集零方差)")
for it in frozen[:30]: print("    idx", it[0], "零方差维数=", it[1])

if problems or extreme or frozen:
    drop = set()
    for k, v in problems.items(): drop |= {it[0] for it in v}
    drop |= {it[0] for it in extreme}
    drop |= {it[0] for it in frozen}
    print("\n[建议剔除]", sorted(drop))
else:
    print("\n[结论] 未发现数值/结构异常 → 无可剔(几何对错需回放,不在本筛范围)")
