#!/bin/bash
# 对**一个** checkpoint 跑完 probe_index_channel4 的五个条件，然后关服务器。
#
# 用法: probe_pass.sh <tag> <config> <exp> <step>
#   tag    结果文件名标签，如 base_d22_19000 / d22idx3k
#   config/exp/step  指向 checkpoints/<config>/<exp>/<step>
#
# 五个条件（前三个量索引块，第四个是真空对照，第五个量视觉）：
#   all  ref  整块索引 18:22，图像真值
#   idx  ref  只脏 cycle_idx(21)
#   label ref 只脏 goal_y/cycle_dist(19,20)   ← **pilot(d22idx) 训练构造的就是这个条件**
#   none ref  真空对照：两路输入逐字节相同，proj 必须落在 0±噪声内
#   none don  只把图像换成供体集的，量"视觉单独"的拉力（判决实验）
#   label don 脏 19/20 **并且**换图  ← **改法 A(d22idxI) 训练构造的就是这个条件**
#
# 主指标是 **proj（mm）**，不是 frac：proj 是"交换后的动作位移在参照轨迹方向上的
# 有符号投影"，零假设恰好为 0；frac 是个比值，分母（轨迹差）太小的对会把噪声放大。
# frac 现在只作附注打印（probe v4 起）。
set -euo pipefail

TAG="${1:?用法: probe_pass.sh <tag> <config> <exp> <step>}"
CONFIG="$2"; EXP="$3"; STEP="$4"
ROOT="/home/dgl/lxc/openpi"
PROBE="/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker"
PY="/home/dgl/lxc/openpi/.venv/bin/python"
LOGS="/home/dgl/lxc/.runlogs"
EPISODES="${EPISODES:-3}"
REPS="${REPS:-5}"

DIR="$ROOT/checkpoints/$CONFIG/$EXP/$STEP"
[ -d "$DIR" ] || { echo "✗ 存档不存在: $DIR"; exit 1; }
if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 有训练在跑，GPU 塞不下"; exit 1
fi

echo "=== $(date '+%T') 起服务器：$DIR ==="
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
nohup .venv/bin/python scripts/serve_policy.py --port 8010 policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$DIR" \
    > "$LOGS/serve_${TAG}.log" 2>&1 &
SPID=$!
for _ in $(seq 1 150); do
    grep -q "server listening" "$LOGS/serve_${TAG}.log" 2>/dev/null && break
    if ! ps -p "$SPID" >/dev/null 2>&1; then
        echo "✗ 服务器退出："; tail -20 "$LOGS/serve_${TAG}.log"; exit 1
    fi
    sleep 4
done
grep -q "server listening" "$LOGS/serve_${TAG}.log" || { echo "✗ 服务器超时"; exit 1; }
echo "=== $(date '+%T') 服务器就绪，开始五个条件 ==="

cd "$PROBE"
for SPEC in "all ref" "idx ref" "label ref" "none ref" "none don" "label don"; do
    # shellcheck disable=SC2086
    set -- $SPEC
    echo "----- $(date '+%T') dims=$1 image=$2 -----"
    # 过滤串必须跟着指标一起改：v4 起主指标是 **proj**，小结行也换了格式，
    # 旧的 "平均 frac|frac>0" 一个都匹配不上 ⇒ 日志里整段小结消失（JSON 还是全的）。
    "$PY" probe_index_channel4.py --run --episodes "$EPISODES" --reps "$REPS" \
        --tag "$TAG" --dims "$1" --image "$2" \
        2>&1 | grep -aE "平均 proj|附注 frac|噪声sd 中位|小结|proj [+-][0-9]|cross-ep|Traceback|Error|✗" || true
done

echo "=== $(date '+%T') 关服务器 ==="
kill "$SPID" 2>/dev/null || true
sleep 4
echo "=== $TAG 全部完成 $(date '+%T') ==="
