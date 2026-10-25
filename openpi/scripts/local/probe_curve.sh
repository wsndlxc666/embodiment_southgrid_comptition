#!/bin/bash
# 沿存档序列跑反事实探针，得到"tgt_rel 扰动量 vs 训练步数"曲线。
#
# 为什么要曲线而不是单点（docs/stage2_restart_20260922.md §5.1）：
#   "模型没学会用 tgt_rel" 和 "模型还没训够" 在单点上长得一模一样。
#   沿存档序列看：扰动量随步数上升 -> 通道真的在被学；一直平在噪声级 -> 才是无视它。
#
# 为什么要等训练退出：探针要加载策略上卡，而训练已占 17.7GB / 24GB。
#
# 用法: probe_curve.sh [deadline_HHMM]
#   默认等训练自然结束；到 deadline 还没结束就 SIGINT 掉（每个 500 步有存档，
#   丢的只是最后一个不满 500 步的尾巴），然后开跑。
set -uo pipefail

DEADLINE="${1:-0610}"
CONFIG="pi05_g1_omnipicker_clear5_stage2_lora"
EXP="clear5_stage2_oracle_181ep"
CKROOT="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
TRAIN_LOG="/home/dgl/lxc/.runlogs/train_stage2.log"
OUT="/home/dgl/lxc/.runlogs/probe_curve_tgt_rel.txt"
PIDFILE="/tmp/stage2.pid"
PY=/home/dgl/lxc/openpi/.venv/bin/python

# deadline 是"下一个 HH:MM"。**必须用 epoch 比较**，否则同一天比较会在
# 2355 >= 0645 时立刻成立 —— 那会把正在跑的训练当场掐掉（踩过这个坑）。
DL_H=${DEADLINE:0:2}; DL_M=${DEADLINE:2:2}
DL_EPOCH=$(date -d "today ${DL_H}:${DL_M}" +%s)
[ "$DL_EPOCH" -le "$(date +%s)" ] && DL_EPOCH=$(date -d "tomorrow ${DL_H}:${DL_M}" +%s)
echo "deadline=${DEADLINE} -> $(date -d "@$DL_EPOCH" '+%F %T')（epoch $DL_EPOCH）"

mkdir -p /home/dgl/lxc/.runlogs

{
echo "===== 探针流水线启动 $(date '+%F %T')，deadline=${DEADLINE} ====="

# ---- 1. 等训练结束（自然退出，或到点掐掉）----
PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "[1] 训练在跑 (PID $PID)，等它结束或到 ${DEADLINE}..."
    while ps -p "$PID" >/dev/null 2>&1; do
        if [ "$(date +%s)" -ge "$DL_EPOCH" ]; then
            echo "[1] 到点 ${DEADLINE}，SIGINT 训练（存档每 500 步，只丢尾巴）"
            kill -INT "$PID" 2>/dev/null
            # 给它 120s 收尾；还在就 SIGTERM
            for _ in $(seq 1 24); do
                ps -p "$PID" >/dev/null 2>&1 || break
                sleep 5
            done
            ps -p "$PID" >/dev/null 2>&1 && { echo "[1] SIGINT 无响应，SIGTERM"; kill -TERM "$PID" 2>/dev/null; sleep 20; }
            break
        fi
        sleep 30
    done
    ps -p "$PID" >/dev/null 2>&1 && echo "[1] ⚠ 训练进程仍在（可能没杀干净）" || echo "[1] ✓ 训练已退出"
else
    echo "[1] 训练进程不在（可能已结束）"
fi

# 等显存真的释放（JAX 退出后驱动回收有延迟）
for _ in $(seq 1 30); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free_mib" -gt 12000 ] && break
    sleep 10
done
echo "[1] 空闲显存 ${free_mib} MiB"

# ---- 2. 列出存档点（抽稀）----
# save_interval=500 -> 满训有 24 个点，全跑要 ~36 分钟，且 500/1000 步的点
# 还几乎没训到，信息量低。取 2000 的整数倍 + 最后一个（无论落在哪）。
mapfile -t ALL < <(ls "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
if [ "${#ALL[@]}" -eq 0 ]; then
    echo "[2] ✗ 一个存档点都没有，放弃"; exit 1
fi
STEPS=()
for s in "${ALL[@]}"; do
    [ "$((s % 2000))" -eq 0 ] && STEPS+=("$s")
done
last="${ALL[-1]}"
[ "${STEPS[-1]:-}" != "$last" ] && STEPS+=("$last")
echo "[2] 存档点共 ${#ALL[@]} 个 (${ALL[0]}..$last)，抽稀后探 ${#STEPS[@]} 个: ${STEPS[*]}"

# ---- 3. 逐点探针 ----
echo
echo "===== 反事实探针：tgt_rel 扰动量 vs 步数 $(date '+%F %T') ====="
printf '%-8s %-14s %-14s %-12s %s\n' "步数" "other扰动量" "zero扰动量" "自然变化" "判读"
echo "--------------------------------------------------------------------------"

for s in "${STEPS[@]}"; do
    log="/tmp/probe_${s}.log"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$PY" \
        /home/dgl/lxc/openpi/scripts/local/probe_tgt_rel_usage.py \
        "$CKROOT/$s" 0 "$CONFIG" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        printf '%-8s  ✗ 探针失败 (rc=%d)，见 %s\n' "$s" "$rc" "$log"
        continue
    fi
    other=$(grep -oE "干预 other: 中位 [0-9.]+" "$log" | head -1 | grep -oE "[0-9.]+$")
    zero=$(grep -oE "干预 zero : 中位 [0-9.]+" "$log" | head -1 | grep -oE "[0-9.]+$")
    nat=$(grep -oE "自然变化 中位 [0-9.]+" "$log" | head -1 | grep -oE "[0-9.]+$")
    # 判读：other/自然变化 的比例
    verdict=$(awk -v o="$other" -v n="$nat" 'BEGIN{
        if (n<=0 || o=="") {print "?"; exit}
        r=o/n
        if (r<0.05) printf "无视 (%.1f%%)", r*100
        else if (r<0.5) printf "很弱 (%.1f%%)", r*100
        else printf "在用 (%.1f%%)", r*100
    }')
    printf '%-8s %-14s %-14s %-12s %s\n' "$s" "${other:-?}" "${zero:-?}" "${nat:-?}" "$verdict"
done

echo
echo "===== 判读（判据事先写死在 probe_tgt_rel_usage.py 里）====="
echo "  扰动量随步数**单调上升** -> tgt_rel 通道真的在被学，值得投入 perception"
echo "  全程平在自然变化的 5% 以下 -> 模型无视它；oracle 都无效 => depth/vision 估位只会更差，整条可砍"
echo
echo "原始日志: /tmp/probe_<步数>.log"
echo "===== 完成 $(date '+%F %T') ====="
} 2>&1 | tee "$OUT"
