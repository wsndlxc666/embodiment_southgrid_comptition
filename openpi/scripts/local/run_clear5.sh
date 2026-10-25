#!/bin/bash
# clear5 训练启动 + 实时查看 入口(tmux 版,SSH 断开训练照跑)。
# 用法:
#   ~/run_clear5.sh                                  首轮快跑(默认 s1500:1500步 短LR, batch24 ≈ 1.5天)
#   ~/run_clear5.sh clear5_v1 resume                 续训最新存档
#   ~/run_clear5.sh clear5_v2 new 24 3000            改 exp/步数
#   ~/run_clear5.sh clear5_v1 new 24 12000 pi05_g1_omnipicker_clear5_lora   满训(原长LR配置)
#   ~/run_clear5.sh attach                           已有会话在跑时,直接附着(等价 tmux attach -t clear5)
SESSION=clear5
CONFIG_DEFAULT="pi05_g1_omnipicker_clear5_lora_s1500"

# 已在跑:直接附着(避免双开把 GPU 打爆)
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[已有 $SESSION 在跑,直接附着…] 按 Ctrl-b 松开再按 d 可脱离,训练不受影响"
  sleep 1
  tmux attach -t "$SESSION"
  exit 0
fi

if [ "${1:-}" = "attach" ]; then
  echo "当前没有在跑的 $SESSION 会话,先跑: ~/run_clear5.sh"
  exit 1
fi

EXP="${1:-clear5_v1}"
MODE="${2:-new}"
BATCH="${3:-24}"
STEPS="${4:-1500}"
CONFIG="${5:-$CONFIG_DEFAULT}"
LOG="$HOME/train_${EXP}.log"
rm -f "$LOG"   # 本次新日志

tmux new-session -d -s "$SESSION" \
  "bash $HOME/train_launch.sh $EXP $MODE $BATCH $STEPS $CONFIG 2>&1 | tee $LOG"

echo "[已启动] config=$CONFIG exp=$EXP mode=$MODE batch=$BATCH steps=$STEPS  tmux会话=$SESSION"
echo
echo "  实时看训练(终端里直接看,和 ssh 断开无关):"
echo "      tmux attach -t $SESSION       ← 看进度;看完 Ctrl-b 松手再按 d 脱离,后台继续训"
echo "  不附着、只盯日志:"
echo "      tail -f $LOG"
echo "  显存/占用:"
echo "      watch -n2 nvidia-smi"
echo "  停训练 / 彻底重启(重跑前先停):"
echo "      tmux kill-session -t $SESSION"
