#!/bin/bash
# 本机 3090 JAX-LoRA 续训(resume):从上次末步 1499 继续,不加 --overwrite。
# num_train_steps = 新的总步数(循环 range(当前步, num_train_steps))。
# 用法: train_local_3090_jax_resume.sh [exp] [total_steps]
set -euo pipefail
EXP="${1:-clear5_v1_l3090_lora}"
TOTAL="${2:-10000}"
CONFIG="pi05_g1_omnipicker_clear5_lora_s1500"
LOG="/home/dgl/lxc/.runlogs/train_jax_${EXP}_resume.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1

echo "===== clear5 JAX-LoRA 续训启动 $(date '+%F %T') =====" | tee "$LOG"
echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL (resume, 基座+数据均本地缓存)" | tee -a "$LOG"

nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 --num_train_steps "$TOTAL" \
    --resume \
    >> "$LOG" 2>&1 &
echo $! > /tmp/clear5_local_jax_resume.pid
echo "PID $(cat /tmp/clear5_local_jax_resume.pid)  日志: $LOG"
