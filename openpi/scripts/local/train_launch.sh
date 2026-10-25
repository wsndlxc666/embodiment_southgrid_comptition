#!/bin/bash
# 实际训练执行器(在 tmux 会话内运行;也支持直接前台跑)。
# 参数: [exp] [mode=new|resume] [batch] [steps] [config]
#   config 默认 s1500(短LR日程首轮快跑)；满训/续训长程时传 pi05_g1_omnipicker_clear5_lora
set -euo pipefail
EXP="${1:-clear5_v1}"
MODE="${2:-new}"
BATCH="${3:-24}"
STEPS="${4:-1500}"
CONFIG="${5:-pi05_g1_omnipicker_clear5_lora_s1500}"

cd /home/k24lxc/openpi
export HF_LEROBOT_HOME=/home/k24lxc/openpi/data

EXTRA=""
[ "$MODE" = "resume" ] && EXTRA="--resume"

echo "===== clear5 训练启动 $(date '+%F %T') ====="
echo "config=$CONFIG exp=$EXP mode=$MODE batch=$BATCH steps=$STEPS num_workers=8 (机器: $(hostname))"

exec .venv/bin/python scripts/train_pytorch.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size "$BATCH" --num_train_steps "$STEPS" \
    --num_workers 8 --log_interval 10 \
    $EXTRA
