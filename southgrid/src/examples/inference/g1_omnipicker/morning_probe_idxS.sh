#!/bin/bash
# 只跑**机制探针**（六条件），**不需要 OrcaLab 仿真** —— 探针只用数据集 + 策略服务器。
# 所以早上起来第一件能做的事就是它，不必等你在 GUI 里把场景摆好。
#
# 它回答的是"模型有没有开始听相位通道"，与"仿真里收了几件"是两回事，两个都要看。
#
# 对 idxS 这一档，**对照臂 = idxR/11999**（它的热启动源，即"换数据之前"），
# 这样前/后只差数据本身，不差步数、不差 recipe。
#
# ⚠ verdict_probe_v4.py 里写死的三条判据是给 idxI 那轮（判据方向：label_imgdon 该**降**）
#   设计的。idxS 这轮的假设不同，**它的自动判定只能当参考**，数要自己读：
#   · `label_imgdon` 这一档的输入 = 图像取自供体、dims19/20 取自供体、dim21 取参照
#     —— 若模型开始按索引走，这一档的 proj 应当朝 0 靠。
#   · 主指标是**有符号投影** proj（mm），零假设严格为 0；别用范数类指标。
#
# 用法: morning_probe_idxS.sh [--episodes 3] [--reps 5] [--steps 11999]
set -euo pipefail

INF=/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
ROOT=/home/dgl/lxc/openpi
LOGS=/home/dgl/lxc/.runlogs
OPY="$ROOT/.venv/bin/python"

EPISODES=3
REPS=5
STEPS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --episodes) EPISODES="$2"; shift 2 ;;
        --reps)     REPS="$2";     shift 2 ;;
        --steps)    STEPS="$2";    shift 2 ;;
        *) echo "未知参数 $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------ 前置闸
if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练在跑（BFC 预占 ~22.6G），服务器起不来。等它退出。"; exit 1
fi

NEW_CONFIG=pi05_g1_omnipicker_clear5_d22idxS_lora_12k
NEW_EXP=clear5_d22idxS_lora_100ep
NEW_DIR="$ROOT/checkpoints/$NEW_CONFIG/$NEW_EXP"
if [ -z "$STEPS" ]; then
    STEPS=$(ls "$NEW_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
    [ -n "$STEPS" ] || { echo "✗ $NEW_DIR 下没有数字存档目录 —— 训练还没落盘过？"; exit 1; }
    echo "idxS 取最新正式档 $STEPS（可点：$(ls "$NEW_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' '))"
fi
BASE_TAG=d22idxR_11999
BASE_CONFIG=pi05_g1_omnipicker_clear5_d22idxR_lora_12k
BASE_EXP=clear5_d22idxR_lora_100ep
BASE_STEP=11999

SPID=""
cleanup() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null || true; }
trap cleanup EXIT

serve_and_wait() {   # $1=config $2=dir $3=logname → 回显 PID
    local cfg="$1" dir="$2" log="$LOGS/$3"
    if ss -ltn 2>/dev/null | grep -q ':8010'; then
        echo "✗ 8010 已被占用，先停掉它（本脚本只管理自己起的那个）" >&2
        ss -ltnp 2>/dev/null | grep ':8010' >&2 || true
        return 1
    fi
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    nohup "$OPY" scripts/serve_policy.py --port 8010 policy:checkpoint \
        --policy.config="$cfg" --policy.dir="$dir" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 150); do
        grep -q "server listening" "$log" 2>/dev/null && { echo "$pid"; return 0; }
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "✗ 服务器退出：" >&2; tail -20 "$log" >&2; return 1
        fi
        sleep 4
    done
    echo "✗ 服务器 10 分钟没就绪" >&2; return 1
}

probe_six() {   # $1=tag
    local tag="$1"
    cd "$INF"
    for SPEC in "all ref" "idx ref" "label ref" "none ref" "none don" "label don"; do
        # shellcheck disable=SC2086
        set -- $SPEC
        echo "  ----- $(date '+%T') dims=$1 image=$2 -----"
        "$OPY" probe_index_channel4.py --run --episodes "$EPISODES" --reps "$REPS" \
            --tag "$tag" --dims "$1" --image "$2" \
            2>&1 | grep -aE "平均 proj|附注 frac|噪声sd 中位|小结|cross-ep|Traceback|Error|✗" || true
    done
}

run_arm() {   # $1=tag $2=config $3=exp $4=step
    local tag="$1" cfg="$2" exp="$3" step="$4"
    local dir="$ROOT/checkpoints/$cfg/$exp/$step"
    if [ ! -d "$dir" ]; then echo "✗ 跳过 $tag：存档不在 $dir"; return 0; fi
    echo
    echo "================ $(date '+%F %T')  档 = $tag  ($dir) ================"
    SPID=$(serve_and_wait "$cfg" "$dir" "serve_$tag.log")
    echo "  服务器 PID $SPID"
    probe_six "$tag"
    kill "$SPID" 2>/dev/null || true
    SPID=""
    sleep 4
}

run_arm "$BASE_TAG" "$BASE_CONFIG" "$BASE_EXP" "$BASE_STEP"
run_arm "d22idxS_$STEPS" "$NEW_CONFIG" "$NEW_EXP" "$STEPS"

echo
echo "================ $(date '+%F %T') 判决 ================"
cd "$INF"
"$OPY" verdict_probe_v4.py "$BASE_TAG" "d22idxS_$STEPS" || true
echo
"$OPY" compare_probe_v4.py "$BASE_TAG" "d22idxS_$STEPS" || true
echo
echo "读法：主指标是有符号投影 proj（mm），零假设严格为 0。"
echo "  · 新档在 'all ref' / 'idx ref' 上比对照臂**更大** ⇒ 模型确实更听相位通道了。"
echo "  · 'label imgdon' 朝 0 靠 ⇒ 图像与相位打架时它开始听相位。"
echo "  · 'none ref' 是内建自检（两路输入逐字节相同），必须落在 0±噪声内，否则量测本身有问题。"
