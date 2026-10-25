#!/bin/bash
# 训练完之后一条命令做完：起服务器 → 重量四变体 → 配对比较 → 关服务器。
#
# 用法: after_train_probe.sh <tag> [exp_name] [config]
#   tag      结果文件名后缀，如 d22idx3k
#   exp_name 默认 clear5_d22idx_lora_100ep_pilot3k
#   config   默认 pi05_g1_omnipicker_clear5_d22idx_lora_20k
#
# 注意：训练进程必须已经退出，否则 GPU 塞不下（训练 ~19G / 24G）。
#
# 会跑两轮服务器：
#   第 1 轮 d22 基线 —— 只为补测 (dims=none, image=don) 这个镜像条件
#                        （all/idx/label 三个基线已经有了，跳过省 18 分钟）
#   第 2 轮 新档     —— 四个条件全测
set -euo pipefail

TAG="${1:?用法: after_train_probe.sh <tag> [exp_name] [config]}"
EXP="${2:-clear5_d22idx_lora_100ep_pilot3k}"
CONFIG="${3:-pi05_g1_omnipicker_clear5_d22idx_lora_20k}"
ROOT="/home/dgl/lxc/openpi"
PROBE="/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker"
LOGS="/home/dgl/lxc/.runlogs"
PY="/home/dgl/lxc/openpi/.venv/bin/python"
EPISODES="${EPISODES:-3}"
REPS="${REPS:-6}"
BASE_TAG="base_d22_19000"
BASE_CONFIG="pi05_g1_omnipicker_clear5_d22_lora_20k"
BASE_EXP="clear5_d22_lora_100ep"
BASE_STEP="19000"

if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练进程在跑，先等它退出（GPU 塞不下）"; exit 1
fi

# serve_and_wait <config> <dir> <logname> —— 起服务器并等就绪，回显 PID
serve_and_wait() {
    local cfg="$1" dir="$2" log="$3"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    nohup .venv/bin/python scripts/serve_policy.py --port 8010 policy:checkpoint \
        --policy.config="$cfg" --policy.dir="$dir" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 150); do
        grep -q "server listening" "$log" 2>/dev/null && { echo "$pid"; return 0; }
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "✗ 服务器进程退出：" >&2; tail -20 "$log" >&2; return 1
        fi
        sleep 4
    done
    echo "✗ 服务器 10 分钟没就绪" >&2; return 1
}

# run_spec <tag> <dims> <image>
run_spec() {
    cd "$PROBE"
    echo "----- $(date '+%T') tag=$1 dims=$2 image=$3 -----"
    "$PY" probe_index_channel3.py --run --episodes "$EPISODES" --reps "$REPS" \
        --tag "$1" --dims "$2" --image "$3" \
        2>&1 | grep -aE "平均斜率|加权点估计|小结|噪声sd|Traceback|Error" || true
}

# ---------- 第 1 轮：补基线里缺的那个镜像条件 ----------
if [ ! -f "$PROBE/probe_v3_${BASE_TAG}_none_imgdon.json" ]; then
    echo "=== 第 1 轮：d22 基线补测 (none, don) ==="
    SPID=$(serve_and_wait "$BASE_CONFIG" "$ROOT/checkpoints/$BASE_CONFIG/$BASE_EXP/$BASE_STEP" \
                          "$LOGS/serve_${BASE_TAG}_imgdon.log")
    run_spec "$BASE_TAG" none don
    kill "$SPID" 2>/dev/null || true
    sleep 4
else
    echo "=== 基线镜像条件已存在，跳过第 1 轮 ==="
fi

# ---------- 第 2 轮：新档 ----------
CK="$ROOT/checkpoints/$CONFIG/$EXP"
STEP=$(ls "$CK" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
[ -n "$STEP" ] || { echo "✗ $CK 下没有数字存档目录"; exit 1; }
echo "=== 第 2 轮：新档 $CK/$STEP ==="
SPID=$(serve_and_wait "$CONFIG" "$CK/$STEP" "$LOGS/serve_${TAG}.log")
for SPEC in "all ref" "idx ref" "label ref" "none don"; do
    # shellcheck disable=SC2086
    run_spec "$TAG" $SPEC
done
kill "$SPID" 2>/dev/null || true
sleep 3

echo "=== 配对比较 ==="
"$PY" "$PROBE/compare_probe_v3.py" "$BASE_TAG" "$TAG"
echo "=== 全部完成 $(date '+%T') ==="
