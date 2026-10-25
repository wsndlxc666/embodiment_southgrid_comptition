#!/bin/bash
# 自动续采 g1_omnipicker_clear5 直到 1000 集。
# 采集进程崩溃/正常退出后，若未达目标则 sleep 后 --resume 自动重启（可跨 Xid79 等中断）。
# 用法: nohup bash run_clear5_to1000.sh > /dev/null 2>&1 &
set -u
PY=/home/dgl/miniconda3/envs/orcalab/bin/python
SCRIPT=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker/g1_omnipicker_collect_clear5_lerobot.py
WORKDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
DATA_META=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5/meta/episodes.jsonl
LOG=/tmp/clear5_c1000.log
TARGET=1000
cd "$WORKDIR" || exit 1

count_eps() {
    if [ -f "$DATA_META" ]; then
        grep -c '^' "$DATA_META"
    else
        echo 0
    fi
}

while :; do
    n=$(count_eps)
    echo "[watchdog] $(date '+%F %T') episodes=$n target=$TARGET" >> "$LOG"
    if [ "$n" -ge "$TARGET" ]; then
        echo "[watchdog] $(date '+%F %T') 已达到目标 $TARGET 集，退出。" >> "$LOG"
        break
    fi
    echo "[watchdog] $(date '+%F %T') 启动采集器 (resume, num_episodes=$TARGET, fix4: dz=-12 settle=0 probe)..." >> "$LOG"
    "$PY" "$SCRIPT" \
        --resume \
        --num_episodes "$TARGET" \
        --grasp_dz_mm -12 \
        --settle_tol_mm 0 \
        --grasp_probe \
        >> "$LOG" 2>&1
    rc=$?
    echo "[watchdog] $(date '+%F %T') 采集器退出 rc=$rc" >> "$LOG"
    # 检查 sim 是否仍可达，若不可达多等几轮
    if ! timeout 3 bash -c 'cat < /dev/null > /dev/tcp/localhost/50051' 2>/dev/null; then
        echo "[watchdog] $(date '+%F %T') sim(50051) 不可达，等待 60s" >> "$LOG"
        sleep 60
    else
        sleep 20
    fi
done
