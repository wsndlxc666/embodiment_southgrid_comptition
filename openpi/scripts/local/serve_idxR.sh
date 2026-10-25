#!/bin/bash
# 起 idxR 的策略服务器。**只需要给一个步数**（默认 11999）。
#
# 为什么要有这个脚本：`--policy.dir` 要三层拼起来，而 config 名结尾是 `lora_12k`、
# exp 名结尾是 `lora_100ep` —— **两个不一样**，中间还夹一层 exp 目录。
# 2026-09-20 手打时拼成了 `checkpoints/pi05_g1_omnipicker_clear5_d22idxR_lora_100ep/11999`
# （少了中间那层），于是 FileNotFoundError。路径拼装封进脚本里就不会再错。
#
# 正确形状（三层）：
#   checkpoints/<config>/<exp>/<step>
#   checkpoints/pi05_g1_omnipicker_clear5_d22idxR_lora_12k/clear5_d22idxR_lora_100ep/11999
#
# 用法:
#   serve_idxR.sh            # 起 11999（最终档）
#   serve_idxR.sh 10000      # 泛指正式档 5000 / 10000 / 11999
#   serve_idxR.sh 3000       # 中间档自动回退到 _keep_idxR 的硬链接留存
#   PORT=8011 serve_idxR.sh  # 换端口（默认 8010）
set -euo pipefail

STEP="${1:-11999}"
CONFIG="pi05_g1_omnipicker_clear5_d22idxR_lora_12k"
EXP="clear5_d22idxR_lora_100ep"
PORT="${PORT:-8010}"
ROOT=/home/dgl/lxc/openpi
LOG="/home/dgl/lxc/.runlogs/serve_idxR_${STEP}.log"

cd "$ROOT"

# ---- 解析存档路径（正式目录优先，其次硬链接留存）----
DIR="$ROOT/checkpoints/$CONFIG/$EXP/$STEP"
if [ ! -d "$DIR" ]; then
    KEEP="$ROOT/checkpoints/_keep_idxR/$STEP"
    if [ -d "$KEEP" ]; then
        DIR="$KEEP"
    else
        echo "✗ 没有步数 $STEP 的存档。可点："
        echo "    正式档: $(ls "$ROOT/checkpoints/$CONFIG/$EXP" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
        echo "    留存档: $(ls "$ROOT/checkpoints/_keep_idxR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
        exit 1
    fi
fi
[ -d "$DIR/params" ] || { echo "✗ $DIR 里没有 params/，存档不完整"; exit 1; }

# ---- 8010 被占就直接说清楚（这是之前栽过的一种失败）----
if ss -ltn 2>/dev/null | grep -q ":$PORT"; then
    echo "✗ 端口 $PORT 已被占用："
    ss -ltnp 2>/dev/null | grep ":$PORT" || true
    echo "  先停掉它，或换端口：PORT=8011 $0 $STEP"
    exit 1
fi

case "$STEP" in
    11999) NOTE="最终档" ;;
    5000|10000) NOTE="正式档（keep_period=5000 永久保留）" ;;
    *) NOTE="留存的硬链接档；注意重启训练时可能被覆盖" ;;
esac

echo "档位  $CONFIG"
echo "      $EXP/$STEP   ($NOTE)"
echo "路径  $DIR"
echo "端口  $PORT      日志 $LOG"
echo "等 'server listening' 出现即就绪（约 1~3 分钟，要加载 8.7G 权重 + JIT）。"
echo "别自己另起一个占 8010 —— 会 bind 失败。"
echo

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
nohup "$ROOT/.venv/bin/python" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$DIR" \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "/tmp/serve_idxR_${STEP}.pid"

for _ in $(seq 1 150); do
    if grep -q "server listening" "$LOG" 2>/dev/null; then
        echo "✓ 就绪（PID $PID）: $(grep -m1 'server listening' "$LOG")"
        exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "✗ 服务器进程已退出，日志尾部：" >&2
        tail -25 "$LOG" >&2
        exit 1
    fi
    sleep 4
done
echo "✗ 10 分钟没就绪（PID $PID 还活着，自己看 $LOG）" >&2
exit 1
