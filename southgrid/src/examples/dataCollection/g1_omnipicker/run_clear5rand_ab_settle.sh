#!/bin/bash
# A/B：验证「初始位姿抖动 → 空抓」是不是被 --settle_tol_mm 0 放大的。
#
# 背景（实测）：
#   基线 run_clear5_to1000.sh 用的是 --settle_tol_mm 0 且没有初始抖动，201 集跑通。
#   加了 --init_jitter_deg 6 后，冒烟 6 次尝试只成功 1 集（尝试1/3/4/5 全挂在
#   "未离槽" 抬升校验），probe 显示闭爪瞬间末端 y 距工具 10~13mm ——
#   而成功那次只差 2mm。
#   代码里 `--settle_tol_mm > 0` 才是"末端到位才放行闭爪"的闸（默认 8mm），
#   置 0 等于把这闸彻底关掉。所以抖动带来的暂态误差直接变成空抓。
#
# 本脚本就是要证伪/证实这一点：只开 settle 闸，其余不动。
#   --settle_max_extra 60：把等待封顶在 60 步(3s)，避免大量"冻结帧"混进数据；
#                          同时 [settle] 打点会告诉我们真实等了多少步。
# 数据直接写进正式数据集 g1_omnipicker_clear5_rand —— 参数只影响"抓没抓住"，
# 不影响被记录轨迹的内容，所以这 3 集本身就是合用的正式数据。
set -u
PY=/home/dgl/miniconda3/envs/orcalab/bin/python
WORKDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
SCRIPT=$WORKDIR/g1_omnipicker_collect_clear5_lerobot.py
DATA_ROOT=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_rand
LOG=/tmp/clear5rand_ab.log

cd "$WORKDIR" || exit 1
echo "[ab] $(date '+%F %T') 开始（settle_tol_mm=8, max_extra=60, jitter=6）" > "$LOG"

"$PY" "$SCRIPT" \
    --num_episodes 3 \
    --max_attempts 8 \
    --lerobot_out "$DATA_ROOT" \
    --repo_id "local/g1_omnipicker_clear5_rand" \
    --grasp_dz_mm -12 \
    --settle_tol_mm 8 \
    --settle_max_extra 60 \
    --grasp_probe \
    --k_min 1 --k_max 5 \
    --plant_mode mixed --plant_all_frac 0.5 \
    --init_jitter_deg 6 \
    >> "$LOG" 2>&1
echo "[ab] $(date '+%F %T') 退出 rc=$?" >> "$LOG"
