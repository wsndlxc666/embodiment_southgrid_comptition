#!/bin/bash
# 冒烟：随机化采集器跑 3 集，验证 ①②③④ + tool_id 列端到端可用。
# 输出到 g1_omnipicker_clear5_rand_smoke（**不动** g1_omnipicker_clear5）。
set -u
PY=/home/dgl/miniconda3/envs/orcalab/bin/python
WORKDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
SCRIPT=$WORKDIR/g1_omnipicker_collect_clear5_lerobot.py
OUT=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_rand_smoke
LOG=/tmp/clear5rand_smoke.log
cd "$WORKDIR" || exit 1

echo "[smoke] $(date '+%F %T') 开始" > "$LOG"
"$PY" "$SCRIPT" \
    --lerobot_out "$OUT" \
    --repo_id "local/g1_omnipicker_clear5_rand_smoke" \
    --num_episodes 3 \
    --max_attempts 6 \
    --grasp_dz_mm -12 \
    --settle_tol_mm 0 \
    --grasp_probe \
    --k_min 1 --k_max 5 \
    --plant_mode rand \
    --init_jitter_deg 6 \
    >> "$LOG" 2>&1
echo "[smoke] $(date '+%F %T') 退出 rc=$?" >> "$LOG"
