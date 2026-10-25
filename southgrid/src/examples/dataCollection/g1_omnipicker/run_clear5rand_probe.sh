#!/bin/bash
# 良率对照实验：只变「初始抖动」与「闭爪前的驻留」，其余一律不动。
#
# 背景：冒烟 13 次尝试只入 3 集（23%），失败几乎全是"未离槽"空抓，
#       其中 电工刀(右) 9 次里挂 8 次。probe 显示失败那几次闭爪瞬间
#       末端比工具高约 20mm、y 偏 10mm。基线 201 集是 0 抖动、且从没出过这问题。
#
# 用法: TAG=j0 JITTER=0 TOL=0 MAXEXTRA=0 bash run_clear5rand_probe.sh
#   TAG      日志/数据集后缀
#   JITTER   --init_jitter_deg
#   TOL      --settle_tol_mm   (0 = 关掉"到位才闭爪"闸，即基线行为)
#   MAXEXTRA --settle_max_extra (TOL>0 时驻留上限，步；20 步 = 1s)
#   EPS      跑几集
set -u
PY=/home/dgl/miniconda3/envs/orcalab/bin/python
WORKDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
SCRIPT=$WORKDIR/g1_omnipicker_collect_clear5_lerobot.py

TAG=${TAG:?需要 TAG}
JITTER=${JITTER:-0}
TOL=${TOL:-0}
MAXEXTRA=${MAXEXTRA:-0}
EPS=${EPS:-3}

OUT=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_rand_probe_$TAG
LOG=/tmp/clear5rand_probe_$TAG.log

cd "$WORKDIR" || exit 1
echo "[probe:$TAG] $(date '+%F %T') jitter=$JITTER tol=$TOL max_extra=$MAXEXTRA eps=$EPS" > "$LOG"

"$PY" "$SCRIPT" \
    --num_episodes "$EPS" \
    --max_attempts 20 \
    --lerobot_out "$OUT" \
    --repo_id "local/g1_omnipicker_clear5_rand_probe_$TAG" \
    --grasp_dz_mm -12 \
    --settle_tol_mm "$TOL" \
    --settle_max_extra "$MAXEXTRA" \
    --grasp_probe \
    --k_min 1 --k_max 5 \
    --plant_mode mixed --plant_all_frac 0.5 \
    --init_jitter_deg "$JITTER" \
    >> "$LOG" 2>&1
echo "[probe:$TAG] $(date '+%F %T') 退出 rc=$?" >> "$LOG"
