#!/bin/bash
# d22idx: 在 d22 之上只加**数据侧干预**，专门把 cycle_idx 的指向训出来。
#
# 与 train_d22.sh 的区别（config 内部的差异见 config.py 那段注释）：
#   - config/exp 换成 d22idx 那套
#   - 热启动从 **d22 自己的 19000** 接上（d22 那档是从 d19goal 9000 起的）
#   - 数据侧开了 cycle_boundary_window=60 / weight=10 / counterfactual_prob=0.5
#   - 归一化显式复用 d22 那一档的 norm_stats（assets 指向它），保证尺度逐位相同
#
# 流程建议：
#   train_d22idx.sh smoke      # 20 步，确认数据/归一化/热启动/增强全通
#   train_d22idx.sh            # 正式（默认吃 config 里的 3000）
#   train_d22idx.sh 20000      # 觉得有戏再跑长
#
# 用法: train_d22idx.sh [smoke | <total_steps>]
set -euo pipefail

MODE="${1:-3000}"
CONFIG="pi05_g1_omnipicker_clear5_d22idx_lora_20k"
EXP="clear5_d22idx_lora_100ep_pilot3k"
LOG="/home/dgl/lxc/.runlogs/train_d22idx_100ep.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
mkdir -p /home/dgl/lxc/.runlogs

case "$MODE" in
  smoke)
    TOTAL=20; EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_d22idx_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22idx_smoke.log"
    echo "===== d22idx 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== d22idx 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE" | tee -a "$LOG"
echo "热启动源: checkpoints/pi05_g1_omnipicker_clear5_d22_lora_20k/clear5_d22_lora_100ep/19000/params" | tee -a "$LOG"
echo "现有存档: $(ls /home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}/ 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"

# openpi 只看目录是否存在，空目录会让下次启动直接 FileExistsError。只清完全空的。
CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留)" | tee -a "$LOG"
fi

# 必须把 num_train_steps 和 decay_steps 一起传：只改步数不改 decay_steps 的话，
# 跑超过 config 里那个 decay_steps 之后 LR 会一直贴在地板上。
nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d22idx.pid
echo "PID $(cat /tmp/d22idx.pid)"
echo "日志: $LOG"
echo "盯这行: tail -f $LOG"
