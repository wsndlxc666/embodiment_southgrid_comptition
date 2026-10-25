#!/bin/bash
# 本机 3090 训练执行器(nohup 后台,日志见 $LOG,与 DGX 的 run_clear5.sh 平级)。
# 用法: train_local_3090.sh [exp] [batch] [steps]
set -euo pipefail
EXP="${1:-clear5_v1_l3090}"
BATCH="${2:-4}"
STEPS="${3:-1500}"
CONFIG="pi05_g1_omnipicker_clear5_lora_s1500"
LOG="/home/dgl/lxc/.runlogs/train_${EXP}.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data

echo "===== clear5 本机训练启动 $(date '+%F %T') =====" | tee "$LOG"
echo "config=$CONFIG exp=$EXP batch=$BATCH steps=$STEPS num_workers=8 (机器: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader))" | tee -a "$LOG"

nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_pytorch.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size "$BATCH" --num_train_steps "$STEPS" \
    --num_workers 8 --log_interval 10 \
    >> "$LOG" 2>&1 &
echo $! > /tmp/clear5_local_train.pid
echo "PID $(cat /tmp/clear5_local_train.pid)  日志: $LOG"
