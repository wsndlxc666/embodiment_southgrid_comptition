#!/bin/bash
# d22-cycle(state 22 维 = d19_goal 的 19 维 + goal_y / cycle_dist / cycle_idx) 本机 3090 起训。
#
# 与 train_d19goal.sh 的区别：
#   - config/exp 换成 d22 那套, 数据是 g1_omnipicker_clear5_d22_100ep
#   - 那档 config 带 weight_loader 热启动, 从 **d19goal 自己的 9000 步** 接上
#   - 这也是**全新** exp, checkpoint 目录不存在, 所以不加 --resume 也不加 --overwrite
#     (目录不存在时 --overwrite 无事可做; 而一旦有了存档再带 --overwrite 会清空全部)
#
# 先跑冒烟再跑正式的流程建议：
#   train_d22.sh smoke      # 20 步, 只为确认数据/归一化/热启动/loss 能跑通
#   train_d22.sh            # 正式 20000 步(默认)
#   train_d22.sh resume 25000   # 从存档续到 25000
#
# 用法: train_d22.sh [smoke | resume <total_steps> | <total_steps>]
set -euo pipefail

MODE="${1:-20000}"
CONFIG="pi05_g1_omnipicker_clear5_d22_lora_20k"
EXP="clear5_d22_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d22_100ep.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
mkdir -p /home/dgl/lxc/.runlogs

case "$MODE" in
  smoke)
    # 冒烟用**独立 exp 名**：不然会在正式 exp 的 ckpt 目录里留下一个 step-19 的档。
    # warmup 一起降：optax 内部算的是 decay_steps-warmup_steps, warmup=500 配
    # 小 decay_steps 会得到负数直接报错(decay_steps 不传, 吃 config 的 20000)。
    TOTAL=20; EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --overwrite)
    EXP="clear5_d22_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22_smoke.log"
    echo "===== d22-cycle 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  resume)
    TOTAL="${2:?resume 需要给出目标总步数}"; EXTRA=(--num_train_steps "$TOTAL" --resume)
    echo "===== d22-cycle 续训到 $TOTAL $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL")
    echo "===== d22-cycle 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE" | tee -a "$LOG"
echo "热启动源: checkpoints/pi05_g1_omnipicker_clear5_d19goal_lora_20k/clear5_d19goal_lora_100ep/9000/params" | tee -a "$LOG"
echo "现有存档: $(ls /home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}/ 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"

# openpi 只看**目录是否存在**，不看里面有没有档：上次跑崩(或被杀)留下的空目录会让
# 下一次启动直接 FileExistsError。这里只清"完全为空"的目录——有档就一个字节都不碰。
CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留，无档可丢)" | tee -a "$LOG"
fi

# 与 d19goal 同样**不**传 --lr_schedule.decay_steps：让 config 里的 20000 生效。
# 同 step 的 LR 才能对上，逐维 loss 的对比才成立——LR 更低本身就会压低 loss。
nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d22.pid
echo "PID $(cat /tmp/d22.pid)"
echo "日志: $LOG"
echo "盯这行(每 10 步一条, 含逐维 loss): tail -f $LOG"
