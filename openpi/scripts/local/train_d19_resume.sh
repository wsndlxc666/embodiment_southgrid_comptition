#!/bin/bash
# d19(19 维 state+action) 本机 3090 续训。与 train_local_3090_jax_resume.sh 的区别：
# 用的是 d19 专属 config/exp，且**绝不加 --overwrite**(首次启动那句
# "Wiped checkpoint directory" 就是 --overwrite 干的，续训会清空全部存档)。
# num_train_steps = 新的总步数(循环 range(存档步, num_train_steps))。
# 用法: train_d19_resume.sh [total_steps]
set -euo pipefail
TOTAL="${1:-15000}"
CONFIG="pi05_g1_omnipicker_clear5_d19_lora_20k"
EXP="clear5_d19_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d19_100ep.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1

echo "===== d19 续训启动 $(date '+%F %T') =====" | tee -a "$LOG"
echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL (resume, 不加 --overwrite)" | tee -a "$LOG"
echo "现有存档: $(ls /home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}/ 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"

nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 --num_train_steps "$TOTAL" \
    --resume \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d19_resume.pid
echo "PID $(cat /tmp/d19_resume.pid)  日志: $LOG"
