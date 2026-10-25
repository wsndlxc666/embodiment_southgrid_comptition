#!/bin/bash
# 起任务三本轮模型（随机顺序 clear5）的策略服务。
#
# 为什么要有这个脚本：`--policy.dir` 要三层拼起来，而 config 名结尾是 `_lora`、
# exp 名结尾是 `_100ep` —— **两个不一样**，中间还夹一层 exp 目录。手打很容易漏掉
# 中间那层，然后得到 FileNotFoundError。路径拼装封进脚本里就不会再错。
#
# 正确形状（三层）：
#   checkpoints/<config>/<exp>/<step>
#   checkpoints/pi05_g1_omnipicker_clear5_rand_lora/clear5_rand_lora_100ep/11999
#
# 用法:
#   serve_clear5rand.sh              # 起 11999（最终档）
#   serve_clear5rand.sh 10000        # 泛指 5000 / 10000
#   PORT=8011 serve_clear5rand.sh    # 换端口（默认 8010）
#
# 显存说明：本脚本用的 0.55 是为了**与 OrcaLab 仿真共存**（仿真实测常驻约 5.8 GB）。
#   PREALLOCATE=false 时 0.55（13.2 GB）是**上限而非预占**，实测推理服务稳态占用
#   只有 8.28 GB，加仿真共约 14.1 GB。若整卡独占，可以调高。
set -euo pipefail

STEP="${1:-11999}"
CONFIG="pi05_g1_omnipicker_clear5_rand_lora"
EXP="clear5_rand_lora_100ep"
PORT="${PORT:-8010}"
ROOT="${OPENPI_ROOT:-/path/to/openpi}"
LOG="${LOG:-/tmp/serve_clear5rand_${STEP}.log}"

cd "$ROOT"

DIR="$ROOT/checkpoints/$CONFIG/$EXP/$STEP"
if [ ! -d "$DIR" ]; then
    echo "✗ 没有步数 $STEP 的存档。可点："
    echo "    $(ls "$ROOT/checkpoints/$CONFIG/$EXP" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
    exit 1
fi
[ -d "$DIR/params" ] || { echo "✗ $DIR 里没有 params/，存档不完整"; exit 1; }

# 训练在跑的话（BFC 预占整块）服务起不来，直接说清楚
if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练在跑（BFC 预占整块显存），策略服务起不来。"
    exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":$PORT"; then
    echo "✗ 端口 $PORT 已被占用："
    ss -ltnp 2>/dev/null | grep ":$PORT" || true
    exit 1
fi

echo "档位  $CONFIG"
echo "      $EXP/$STEP"
echo "路径  $DIR"
echo "端口  $PORT      日志 $LOG"
echo "等 'server listening' 出现即就绪（约 1~3 分钟，要加载 8.7G 权重 + JIT）。"
echo

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.55}" \
nohup "$ROOT/.venv/bin/python" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$DIR" \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "/tmp/serve_clear5rand_${STEP}.pid"

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
