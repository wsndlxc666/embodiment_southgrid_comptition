#!/bin/bash
# 训完之后一条命令跑完**两件事**：机制诊断（probe v4 六条件）+ 结果度量（闭环自测收几把）。
#
# 为什么两件事都要跑：probe 量的是"图像和 idx 打架时它听谁"（机制），闭环自测量的是
# "仿真里到底收进箱子几把"（结果）。机制对了结果不一定好，反之亦然 —— 两个都看才不会自欺。
#
# 每档只起**一次**服务器（起一次要几分钟 JIT），probe 和自测共用。
#
# 用法: eval_after_d22idxR.sh [--trials N] [--steps "3000 10000 11999"]
#   --trials 闭环自测试数（默认 3；一试用约 6 分钟墙钟）
#   --steps  只跑新档的某几个存档（默认取最新那个 = 11999）
#            正式目录只有 5000/10000/11999（keep_period=5000）；
#            中间档 1000..11000 在 checkpoints/_keep_idxR/ 的硬链接留存里，
#            直接点步数即可，run_arm 会自动回退到那里。
#
# 前档固定 = **idxP/5000**，即 idxR 热启动的那个源，这样"前/后"只差训练本身，
# 不差数据、不差参照点常量、不差迭代次数。注意别拿 idxP/3000 当前档去比 ——
# 那是用户在评分器里跑过的那个（rollout_idxP_3000.jsonl，1/5），少 2000 步，
# 混进来就把"训练带来的变化"和"多看 2000 步带来的变化"搅在一起了。
set -euo pipefail

ROOT=/home/dgl/lxc/openpi
INF=/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
EVALDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
LOGS=/home/dgl/lxc/.runlogs
OPY="$ROOT/.venv/bin/python"                    # openpi 侧：起服务器、跑 probe
CPY=/home/dgl/miniconda3/envs/orcalab/bin/python # 仿真侧：闭环自测（orca_gym 在这个 env）
REFS="$INF/tool_refs_d22.json"                   # **必须与训练同档**，否则 state 维数对不上

BASE_TAG=d22idxP_5000
BASE_CONFIG=pi05_g1_omnipicker_clear5_d22idxP_lora_20k
BASE_EXP=clear5_d22idxP_lora_100ep
BASE_STEP=5000

NEW_CONFIG=pi05_g1_omnipicker_clear5_d22idxR_lora_12k
NEW_EXP=clear5_d22idxR_lora_100ep
# 留存档（硬链接）目录：keep_period=5000 ⇒ 正式目录只有 5000/10000/11999，
# 中间档（1000..11000）在 snapshot_ckpts.sh 留的 _keep_idxR 里，可随时点。
KEEP="$ROOT/checkpoints/_keep_idxR"

TRIALS=3
STEPS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --trials) TRIALS="$2"; shift 2 ;;
        --steps)  STEPS="$2";  shift 2 ;;
        *) echo "未知参数 $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------ 前置闸
if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练在跑，GPU 塞不下（训练 ~17G / 24G）。等它退出再跑这个。"; exit 1
fi
[ -f "$REFS" ] || { echo "✗ 参照点文件不在：$REFS"; exit 1; }
if ! ss -ltn 2>/dev/null | grep -q ':50051'; then
    echo "✗ 仿真服务 50051 没在听 —— 先起 OrcaLab 仿真，闭环自测连不上"; exit 1
fi

# 新档要跑的存档列表
if [ -z "$STEPS" ]; then
    CK="$ROOT/checkpoints/$NEW_CONFIG/$NEW_EXP"
    STEPS=$(ls "$CK" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    [ -n "$STEPS" ] || { echo "✗ $CK 下没有数字存档目录"; exit 1; }
    echo "默认取最新正式档 $STEPS"
    echo "  正式档可点: $(ls "$CK" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
    echo "  留存档可点: $(ls "$KEEP" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
fi

SPID=""
cleanup() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null || true; }
trap cleanup EXIT

serve_and_wait() {   # $1=config $2=dir $3=logname  → 回显 PID
    local cfg="$1" dir="$2" log="$LOGS/$3"
    cd "$ROOT"
    echo "  \$ .venv/bin/python scripts/serve_policy.py --port 8010 policy:checkpoint \\"
    echo "        --policy.config=$cfg --policy.dir=$dir"
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

probe_six() {   # $1=tag —— 六个条件，和 probe_pass.sh 完全一样
    local tag="$1"          # 先存下来：下面 `set -- $SPEC` 会把位置参数冲掉
    cd "$INF"
    for SPEC in "all ref" "idx ref" "label ref" "none ref" "none don" "label don"; do
        # shellcheck disable=SC2086
        set -- $SPEC
        echo "  ----- $(date '+%T') dims=$1 image=$2 -----"
        "$OPY" probe_index_channel4.py --run --episodes "${EPISODES:-3}" --reps "${REPS:-5}" \
            --tag "$tag" --dims "$1" --image "$2" \
            2>&1 | grep -aE "平均 proj|附注 frac|噪声sd 中位|小结|cross-ep|Traceback|Error|✗" || true
    done
}

rollout() {   # $1=tag —— 闭环自测：种 5 把 → 策略闭环 → 报入箱几把
    cd "$EVALDIR"
    echo "  \$ python g1_omnipicker_policy_clear_eval.py --port 8010 --trials $TRIALS \\"
    echo "        --sim_budget_s 130 --tool_refs $REFS \\"
    echo "        --dump_chunks $LOGS/rollout_$1.jsonl"
    "$CPY" g1_omnipicker_policy_clear_eval.py --port 8010 --trials "$TRIALS" \
        --sim_budget_s 130 --tool_refs "$REFS" \
        --dump_chunks "$LOGS/rollout_$1.jsonl" \
        2>&1 | grep -aE "维|入箱|自测结束|判定区|Traceback|Error|✗|✓" || true
}

run_arm() {   # $1=tag $2=config $3=exp $4=step
    local tag="$1" cfg="$2" exp="$3" step="$4"
    local dir="$ROOT/checkpoints/$cfg/$exp/$step"
    # 正式目录没有就回退到硬链接留存档（中间步数只能从这里取）
    if [ ! -d "$dir" ] && [ -d "$KEEP/$step" ]; then
        dir="$KEEP/$step"
    fi
    # 用 `if` 而不是 `[ -d ] ||`：本脚本 set -e，直接 return 1 会把整条命令**中止**，
    # 于是"要了一个已经被清掉的步数"这种小事会把后面所有档一起带走。
    if [ ! -d "$dir" ]; then
        echo "✗ 跳过 $tag：存档不在 $dir"
        return 0
    fi
    echo
    echo "================ $(date '+%F %T')  档 = $tag  ($dir) ================"
    SPID=$(serve_and_wait "$cfg" "$dir" "serve_$tag.log")
    echo "  服务器 PID $SPID，开始六条件 + 闭环自测"
    probe_six "$tag"
    rollout "$tag"
    kill "$SPID" 2>/dev/null || true
    SPID=""
    sleep 4
}

# ------------------------------------------------------------------ 主流程
# **存档会被自动清理**：`keep_period=5000` ⇒ orbax 只永久保留 step % 5000 == 0 的那几个
# 外加最新一个，其余（1000/2000/…）在下一个存档落盘时就被删。实测起训 3000 步时目录里
# 只剩 `3000`，1000 和 2000 已经没了。所以 --steps 只能点 **5000 的倍数**，
# 或者干脆先看清楚目录里有什么再点。
run_arm "$BASE_TAG" "$BASE_CONFIG" "$BASE_EXP" "$BASE_STEP"
for s in $STEPS; do
    run_arm "d22idxR_$s" "$NEW_CONFIG" "$NEW_EXP" "$s"
done

echo
echo "================ 机制判决（三条判据在 verdict_probe_v4.py 头部，看数之前就写好了）================"
cd "$INF"
for s in $STEPS; do
    echo
    "$OPY" verdict_probe_v4.py "$BASE_TAG" "d22idxR_$s" || true
    echo
    "$OPY" compare_probe_v4.py "$BASE_TAG" "d22idxR_$s" || true
done

echo
echo "================ 闭环自测复核（不看 [已入箱] 那个会计错的计数）================"
# 两件事要分开看，它们指向完全不同的修法：
#   ⑤ check_final_positions  终局坐标 → 工具"没动 / 被拖到别处 / 真的在箱里"
#   ③ analyze_chunks         每次推理想去的位姿 → "最近规划距"是它**够不够得着**那把工具
# 实测基线（2/5 那一轮）：5 把都瞄过，但扳手/螺丝刀最近规划距 27cm（压根没靠近），
# 手电筒/电工刀(左)却能做到 2cm —— 即"瞄了但到不了"与"到了但没抓成"是两回事，
# 只看入箱数会把这两种失败混成一个数。
for s in $STEPS; do
    D="$LOGS/rollout_d22idxR_$s.jsonl"
    [ -f "$D" ] || { echo "（没有 $D，跳过复核）"; continue; }
    echo
    echo "----- d22idxR_$s 终局坐标 -----"
    "$OPY" "$INF/check_final_positions.py" "$D" || true
    echo "----- d22idxR_$s 每次尝试瞄了哪把 / 最近规划距 -----"
    "$OPY" "$INF/analyze_chunks.py" "$D" 2>&1 \
        | grep -aE "推理次数|执行模式|尝试[0-9]+:|共 [0-9]+ 次尝试" || true
done

echo
echo "================ $(date '+%F %T') 全部完成 ================"
