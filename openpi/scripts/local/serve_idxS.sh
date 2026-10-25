#!/bin/bash
# 起 idxS（相位重排档）的策略服务器。**只需要给一个步数**。
#
# 与 serve_idxR.sh 同形，只有 CONFIG/EXP 不同 —— 路径是三层
#   checkpoints/<config>/<exp>/<step>
# 手打很容易漏掉中间那层 exp（2026-09-20 就这么栽过一次 FileNotFoundError）。
#
# ⚠ 训练**没结束前不要起**：训练用 BFC 预占 ~22.6G，服务器起不来。
#   本脚本检测到 train.py 在跑会直接拒绝。
#
# 用法:
#   serve_idxS.sh              # 起最新正式档
#   serve_idxS.sh 12000        # 指定步数
#   PORT=8011 serve_idxS.sh    # 换端口（默认 8010）
set -euo pipefail

CONFIG="pi05_g1_omnipicker_clear5_d22idxS_lora_12k"
EXP="clear5_d22idxS_lora_100ep"
PORT="${PORT:-8010}"
ROOT=/home/dgl/lxc/openpi
CKDIR="$ROOT/checkpoints/$CONFIG/$EXP"

if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练在跑（BFC 预占 ~22.6G），策略服务器起不来。"
    echo "  日志: tail -f /home/dgl/lxc/.runlogs/train_d22idxS.log"
    exit 1
fi

STEP="${1:-}"
if [ -z "$STEP" ]; then
    STEP=$(ls "$CKDIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
    [ -n "$STEP" ] || { echo "✗ $CKDIR 下没有数字存档目录 —— 训练还没落盘过？"; exit 1; }
    echo "未指定步数，取最新档 $STEP（可点：$(ls "$CKDIR" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' '))"
fi

DIR="$CKDIR/$STEP"
if [ ! -d "$DIR" ]; then
    KEEP="$ROOT/checkpoints/_keep_idxS/$STEP"
    if [ -d "$KEEP" ]; then DIR="$KEEP"; else
        echo "✗ 没有步数 $STEP 的存档。可点："
        echo "    正式档: $(ls "$CKDIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
        exit 1
    fi
fi
[ -d "$DIR/params" ] || { echo "✗ $DIR 里没有 params/，存档不完整"; exit 1; }
[ -f "$DIR/assets/g1_omnipicker_clear5_d22_100ep/norm_stats.json" ] || {
    echo "⚠ $DIR/assets 下没找到 norm_stats —— 起服时会报 Normalization stats not found"
    echo "  （config 里 asset_id 指向 g1_omnipicker_clear5_d22_100ep）"; }

if ss -ltn 2>/dev/null | grep -q ":$PORT"; then
    echo "✗ 端口 $PORT 已被占用："
    ss -ltnp 2>/dev/null | grep ":$PORT" || true
    echo "  先停掉它，或换端口：PORT=8011 $0 $STEP"
    exit 1
fi

LOG="/home/dgl/lxc/.runlogs/serve_idxS_${STEP}.log"
echo "档位  $CONFIG"
echo "      $EXP/$STEP"
echo "路径  $DIR"
echo "端口  $PORT      日志 $LOG"
echo "等 'server listening' 出现即就绪（约 1~3 分钟）。"
echo

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
nohup "$ROOT/.venv/bin/python" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$DIR" \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "/tmp/serve_idxS_${STEP}.pid"

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
