#!/bin/bash
# d22 基线 vs d22idx 3k pilot 的 idx 转向测量（probe v4，主指标 = 有符号 proj，mm）。
#
# 用法: run_v4_both.sh [EPISODES] [REPS]      默认 3 / 5
#
# 分三段：
#   0 量尺自检  —— 只跑 (dims=none, image=ref) 的前几对。这个条件下两路输入逐字节相同，
#                  两次采样是同分布独立样本 ⇒ proj 必须落在 0±SE 内。不 ≈0 就说明统计量坏了，
#                  立刻停，别烧后面两小时。（v3 就是死在这一步没做。）
#   1 基线       —— pi05_g1_omnipicker_clear5_d22_lora_20k / step 19000，五个条件
#   2 3k pilot   —— pi05_g1_omnipicker_clear5_d22idx_lora_20k / step 2999，五个条件
#   3 配对比较
set -uo pipefail

EPISODES="${1:-3}"
REPS="${2:-5}"
PROBE="/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker"
PY="/home/dgl/lxc/openpi/.venv/bin/python"
ROOT="/home/dgl/lxc/openpi"
LOGS="/home/dgl/lxc/.runlogs"

BASE_TAG="base_d22_19000"
BASE_CONFIG="pi05_g1_omnipicker_clear5_d22_lora_20k"
BASE_EXP="clear5_d22_lora_100ep"
BASE_STEP="19000"

NEW_TAG="d22idx3k"
NEW_CONFIG="pi05_g1_omnipicker_clear5_d22idx_lora_20k"
NEW_EXP="clear5_d22idx_lora_100ep_pilot3k"
NEW_STEP="2999"

if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 有训练在跑，GPU 塞不下"; exit 1
fi
for d in "$ROOT/checkpoints/$BASE_CONFIG/$BASE_EXP/$BASE_STEP" \
         "$ROOT/checkpoints/$NEW_CONFIG/$NEW_EXP/$NEW_STEP"; do
    [ -d "$d" ] || { echo "✗ 存档不存在: $d"; exit 1; }
done

serve_and_wait() {   # <config> <dir> <log> → 回显 PID
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
            echo "✗ 服务器退出：" >&2; tail -20 "$log" >&2; return 1
        fi
        sleep 4
    done
    echo "✗ 服务器 10 分钟没就绪" >&2; return 1
}

# ---------------- 第 0 段：量尺自检 ----------------
echo "########## $(date '+%T') 第 0 段：量尺自检（真空对照） ##########"
SPID=$(serve_and_wait "$BASE_CONFIG" "$ROOT/checkpoints/$BASE_CONFIG/$BASE_EXP/$BASE_STEP" \
                      "$LOGS/serve_${BASE_TAG}_nullcheck.log") || exit 1
cd "$PROBE"
"$PY" probe_index_channel4.py --run --episodes 1 --reps 2 \
    --tag nullcheck --dims none --image ref 2>&1 | tail -25
kill "$SPID" 2>/dev/null || true
sleep 4

NULLJ="$PROBE/probe_v4_nullcheck_none.json"
[ -f "$NULLJ" ] || { echo "✗ 自检没产出 json，停"; exit 1; }
"$PY" - "$NULLJ" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
ok = True
for cond, s in d["summary"].items():
    t = s["mean_proj_t"]; p = s["mean_proj_mm"]; n = s["n_pairs"]
    good = abs(t) < 2.5
    ok &= good
    print(f"  {cond[:4]:<6} proj {p:+.2f}±{s['mean_proj_se']:.2f}mm  t={t:+.2f}  n={n}  "
          f"{'✓ 落在 0±噪声内' if good else '✗ 偏离 0，统计量有问题'}")
print("自检通过，继续" if ok else "自检不过，停")
sys.exit(0 if ok else 3)
PYEOF
[ $? -eq 0 ] || { echo "✗ 量尺自检失败，终止"; exit 1; }

# ---------------- 第 1、2 段 ----------------
echo "########## $(date '+%T') 第 1 段：基线 $BASE_TAG ##########"
EPISODES="$EPISODES" REPS="$REPS" "$PROBE/probe_pass.sh" \
    "$BASE_TAG" "$BASE_CONFIG" "$BASE_EXP" "$BASE_STEP" || exit 1

echo "########## $(date '+%T') 第 2 段：3k pilot $NEW_TAG ##########"
EPISODES="$EPISODES" REPS="$REPS" "$PROBE/probe_pass.sh" \
    "$NEW_TAG" "$NEW_CONFIG" "$NEW_EXP" "$NEW_STEP" || exit 1

# ---------------- 第 3 段：配对比较 ----------------
echo "########## $(date '+%T') 第 3 段：配对比较 ##########"
"$PY" "$PROBE/compare_probe_v4.py" "$BASE_TAG" "$NEW_TAG"
echo "=== 全部完成 $(date '+%T') ==="
