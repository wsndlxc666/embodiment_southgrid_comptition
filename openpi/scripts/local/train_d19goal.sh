#!/bin/bash
# d19-goal(第19维 = 到最近关键点的距离, 参照点 = 5 工具 + 1 盒子) 本机 3090 起训。
#
# 与 train_d19_resume.sh(v1) 的区别：
#   - config/exp 换成 d19goal 那套, 数据是 g1_omnipicker_clear5_d19_goal_100ep
#   - 这是**全新** exp, checkpoint 目录不存在, 所以不加 --resume 也不加 --overwrite
#     (目录不存在时 --overwrite 无事可做; 而一旦有了存档再带 --overwrite 会清空全部)
#
# 先跑冒烟再跑正式的流程建议：
#   train_d19goal.sh smoke      # 20 步, 只为确认数据/归一化/loss 能跑通
#   train_d19goal.sh            # 正式 20000 步(默认)
#   train_d19goal.sh resume 25000   # 从存档续到 25000
#
# 用法: train_d19goal.sh [smoke | resume <total_steps> | <total_steps>]
set -euo pipefail

MODE="${1:-20000}"
CONFIG="pi05_g1_omnipicker_clear5_d19goal_lora_20k"
EXP="clear5_d19goal_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d19goal_100ep.log"
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
mkdir -p /home/dgl/lxc/.runlogs

case "$MODE" in
  smoke)
    # 冒烟用**独立 exp 名**：不然会在正式 exp 的 ckpt 目录里留下一个 step-19 的档，
    # 之后正式跑虽然不加 --resume 不会被它带偏，但目录里混着废弃档很碍事。
    # warmup 必须一起降：optax 内部算的是 decay_steps-warmup_steps，config 里
    # warmup=500 而 smoke 的 decay_steps=20 会得到 -480 直接报错。
    # --overwrite 让 smoke 可反复跑(smoke 有独立 exp 名, 清掉的是它自己的目录)。
    TOTAL=20; EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --overwrite)
    EXP="clear5_d19goal_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d19goal_smoke.log"
    echo "===== d19-goal 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  resume)
    TOTAL="${2:?resume 需要给出目标总步数}"; EXTRA=(--num_train_steps "$TOTAL" --resume)
    echo "===== d19-goal 续训到 $TOTAL $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL")
    echo "===== d19-goal 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE" | tee -a "$LOG"
echo "现有存档: $(ls /home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}/ 2>/dev/null | tr '\n' ' ')" | tee -a "$LOG"

# openpi 只看**目录是否存在**，不看里面有没有档：上次跑崩(或被杀)留下的空目录会让
# 下一次启动直接 FileExistsError。这里只清"完全为空"的目录——有档就一个字节都不碰，
# 所以不会误删。不带 --overwrite 也正是为了这个：它会把已有档全清空。
CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留，无档可丢)" | tee -a "$LOG"
fi

# 故意**不**传 --lr_schedule.decay_steps：让 config 里的 20000 生效，与 v1 那轮
# (train_d19_resume.sh 也没传，同样吃 config 的 20000)保持一致。同 step 的 LR 才能对上，
# 逐维 loss 的对比才成立——LR 更低本身就会压低 loss，会把改动效果算到 LR 头上。
# 附带好处：以后想续到 20000 步时，日程是连续的，不会中途重新爬坡。
nohup .venv/bin/python scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d19goal.pid
echo "PID $(cat /tmp/d19goal.pid)"
echo "日志: $LOG"
echo "盯这行(每 10 步一条, 含逐维 loss): tail -f $LOG"
