#!/bin/bash
# 早上第一件事：用**现有 idxR/11999 + 修好的 harness** 跑一次闭环自测。
#
# 为什么先跑这个（而不是先跑昨晚新训的档）：本轮真正的第一手修法在 harness ——
# `--progress_gate` 默认已从 official 改回 grip，与训练语义逐帧对齐。这一改**不需要
# 等任何训练**，用手里就有的 idxR/11999 就能验证。新档（idxS）是它的后备。
#
# 前一轮（.runlogs/rollout_idxR_notl.log）栽在哪，这次要盯的就是它有没有变：
#   · official 口径在 t=28.1s 推了一次 idx（模型从没松过手），模型收到训练里不存在
#     的 state ⇒ idx 从 t=85.1s 到 t=292.0s 冻在 3 长达 207s，全是重复微指令。
#   · 换成 grip 后 idx 只能由模型自己开爪推进 ⇒ 它看到的相位演化与训练同分布。
#
# 用法:
#   morning_run_idxR_grip.sh              # 起服务器(如未起) + 跑一次不限时闭环
#   morning_run_idxR_grip.sh --no-serve   # 服务器已在跑，直接自测
set -euo pipefail

INF=/home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
ROOT=/home/dgl/lxc/openpi
LOGS=/home/dgl/lxc/.runlogs
CPY=/home/dgl/miniconda3/envs/orcalab/bin/python
OPY="$ROOT/.venv/bin/python"
TAG=idxR11999_grip

NOSERVE=0
[ "${1:-}" = "--no-serve" ] && NOSERVE=1

# ------------------------------------------------------------------ 前置闸
if pgrep -f "scripts/train[.]py" >/dev/null; then
    echo "✗ 还有训练在跑 —— 它用 BFC 预分配占着 ~22.6G，策略服务器起不来。"
    echo "  训练日志: tail -f $LOGS/train_d22idxS.log"
    echo "  要么等它跑完，要么确认不需要那批权重后自行停掉（PID 在 /tmp/d22idxS.pid）。"
    exit 1
fi
[ -f "$INF/tool_refs_d22.json" ] || { echo "✗ 参照点文件不在：$INF/tool_refs_d22.json"; exit 1; }

# ------------------------------------------------------------------ 策略服务器
if ss -ltn 2>/dev/null | grep -q ':8010'; then
    echo "✓ 8010 已在听，沿用现有服务器"
else
    if [ "$NOSERVE" = "1" ]; then
        echo "✗ 你说了 --no-serve，但 8010 没在听"; exit 1
    fi
    echo "8010 没在听，起 idxR/11999 …"
    "$ROOT/scripts/local/serve_idxR.sh" 11999 || {
        echo "✗ 服务器没起来。日志：$LOGS/serve_idxR_11999.log"; exit 1; }
fi

# ------------------------------------------------------------------ 仿真
if ! ss -ltn 2>/dev/null | grep -q ':50051'; then
    echo
    echo "✗ 仿真服务 50051 没在听 —— 这步**必须你在 OrcaLab GUI 里做**，脚本替不了。"
    echo "  顺序不能反（反了会报 KeyError 'g1_omnipicker_*_pctrl'）："
    echo "    ① 加载场景  ② Publishing scene  ③ 再进运行模式"
    echo "  中途换场景要退出运行模式重进。判定：看 ~/.orcagym/tmp/ 里最新的 xml"
    echo "  是不是 291863 字节那种（含机器人），而不是 221123 字节那种。"
    exit 1
fi
echo "✓ 50051 在听"

# ------------------------------------------------------------------ 闭环自测（不限时）
mkdir -p "$LOGS/task3_frames_$TAG"
cd "$INF"
echo
echo "================ $(date '+%F %T') 闭环自测 $TAG ================"
echo "口径: --no-score 不限时 | --plant_layout nominal（5 件都在桌面标称位，开局无白送分）"
echo "      --progress_gate grip（新默认，与训练逐帧对齐）"
echo "停止: 五件全入箱 或 Ctrl-C（Ctrl-C 也会写出半截记录）"
echo "日志: $LOGS/rollout_$TAG.log"
echo
set +e
"$CPY" run_task3_tool_storage_scored.py \
    --no-score --plant_layout nominal \
    --jsonl "$LOGS/rollout_$TAG.jsonl" \
    --dump_chunks "$LOGS/rollout_$TAG_chunks.jsonl" \
    --frames_dir "$LOGS/task3_frames_$TAG" \
    --orcagym_addr localhost:50051 \
    2>&1 | tee "$LOGS/rollout_$TAG.log"
set -e

# ------------------------------------------------------------------ 事后复核
echo
echo "================ $(date '+%F %T') 复盘 ================"
echo "----- ① 停顿分析：模型在卡住时是否仍在推理（对照上一轮的 207s 冻结）-----"
[ -f "$INF/analyze_stall.py" ] && "$OPY" "$INF/analyze_stall.py" \
    "$LOGS/rollout_$TAG_chunks.jsonl" 2>&1 | tail -40 || echo "（没有 analyze_stall.py，跳过）"

echo
echo "----- ② 每次尝试瞄了哪把 / 最近规划距 -----"
# --tool_refs 必须显式给 d22 那份：analyze_chunks 默认是 tool_refs_d19goal.json（19 维那套），
# 参照点集不同会把"瞄哪把"算错。
[ -f "$INF/analyze_chunks.py" ] && "$OPY" "$INF/analyze_chunks.py" \
    "$LOGS/rollout_$TAG_chunks.jsonl" --tool_refs "$INF/tool_refs_d22.json" 2>&1 \
    | grep -aE "推理次数|执行模式|尝试[0-9]+:|共 [0-9]+ 次尝试" || echo "（跳过）"

echo
echo "----- ③ 终局坐标：工具是没动 / 被拖走 / 真的在箱里 -----"
[ -f "$INF/check_final_positions.py" ] && "$OPY" "$INF/check_final_positions.py" \
    "$LOGS/rollout_$TAG.jsonl" 2>&1 | tail -20 || echo "（跳过）"

echo
echo "----- ④ 相位轨迹（对照上一轮：三次转换、idx 冻在 3）-----"
grep -a "\[cycle\]\|\[stop\]" "$LOGS/rollout_$TAG.log" || true

echo
echo "================ $(date '+%F %T') 完成 ================"
echo "看这三样：入箱几件 / idx 有没有超过 3 / 停顿是不是还在 idx=3 上重复。"
