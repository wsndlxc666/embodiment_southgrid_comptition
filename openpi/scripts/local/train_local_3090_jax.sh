#!/bin/bash
# 本机 3090 JAX-LoRA 训练器(nohup 后台)。这才是"pi05 基座加载 + 只训 LoRA"的路径:
# scripts/train.py 的 _load_weights_and_validate 会从 /home/dgl/.cache/openpi 读缓存基座(12G,已就位)。
# 用法: train_local_3090_jax.sh [exp] [batch] [steps]
set -euo pipefail
EXP="${1:-clear5_v1_l3090_lora}"
BATCH="${2:-4}"
STEPS="${3:-1500}"
CONFIG="pi05_g1_omnipicker_clear5_lora_s1500"
LOG="/home/dgl/lxc/.runlogs/train_jax_${EXP}.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data

echo "===== clear5 JAX-LoRA 本机训练启动 $(date '+%F %T') =====" | tee "$LOG"
echo "config=$CONFIG exp=$EXP batch=$BATCH steps=$STEPS (机器: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader))" | tee -a "$LOG"
echo "基座缓存: $(du -sh /home/dgl/.cache/openpi/openpi-assets/checkpoints/pi05_base/ 2>/dev/null | cut -f1)" | tee -a "$LOG"

nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size "$BATCH" --num_train_steps "$STEPS" \
    --overwrite \
    >> "$LOG" 2>&1 &
echo $! > /tmp/clear5_local_jax.pid
echo "PID $(cat /tmp/clear5_local_jax.pid)  日志: $LOG"
