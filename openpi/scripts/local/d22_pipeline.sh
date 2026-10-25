#!/bin/bash
# d22 串行流水线：等 norm_stats → 冒烟 20 步 → 过了才放正式 20000 步。
#
# 为什么要串行：norm_stats 和训练都吃 GPU，norm_stats 那步实测占 18 GB，训练那步
# XLA 峰值需求 17 GB —— 同时起必然 OOM，而 OOM 的表现是训练进程直接被 kill，
# 日志里只有一个孤零零的 Killed，很容易被当成"模型跑飞了"。
#
# 冒烟闸门（过不了就停，不放正式训练）：
#   1) 日志里出现 Traceback / Error  → 停
#   2) 日志里出现 "loss=" 的正式训练行 → 才算真的跑起来了
#   3) train_diag.jsonl 里最后一行的逐维 loss 里，第 19/20/21 维必须是有限数
#      —— 这三维修的是新增维度，NaN/inf 说明归一化或切片错了，这种错不会让
#      训练崩，只会让新维永远学不动，不查就是白跑 20000 步。
#
# 用法: d22_pipeline.sh            # 全自动
#       d22_pipeline.sh --no-train # 只到冒烟为止
set -uo pipefail

ROOT=/home/dgl/lxc/openpi
NORM="$ROOT/assets/pi05_g1_omnipicker_clear5_d22_lora_20k/g1_omnipicker_clear5_d22_100ep/norm_stats.json"
SMOKE_LOG=/home/dgl/lxc/.runlogs/train_d22_smoke.log
PIPE_LOG=/home/dgl/lxc/.runlogs/d22_pipeline.log
NO_TRAIN="${1:-}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPE_LOG"; }

say "===== d22 流水线启动 ====="

# ---- 1. 等 norm_stats ------------------------------------------------------
# compute_norm_stats 只在**全部**算完之后才写文件，所以"文件存在"就是"算完了"。
# 最多等 3 小时；超时就报错退出，不硬撑。
say "等 norm_stats: $NORM"
for i in $(seq 1 1080); do
    [ -f "$NORM" ] && break
    # 顺带确认算它的进程还活着——进程没了但文件没出来，说明它崩了，别再干等
    if [ "$i" -gt 4 ] && ! pgrep -f "compute_norm_stats.py --config-name=pi05_g1_omnipicker_clear5_d22_lora_20k" >/dev/null; then
        say "**compute_norm_stats 进程已经不在了，但 norm_stats.json 没生成** —— 它崩了。"
        say "去看它自己的输出，别继续。流水线停止。"
        exit 2
    fi
    sleep 10
done
if [ ! -f "$NORM" ]; then say "等 norm_stats 超时(3h)，停止"; exit 3; fi
say "norm_stats 已生成: $(stat -c '%s bytes, %y' "$NORM")"

# ---- 2. 冒烟 ---------------------------------------------------------------
say "--- 跑冒烟(20 步) ---"
: > "$SMOKE_LOG"
bash "$ROOT/scripts/local/train_d22.sh" smoke >/dev/null 2>&1
say "冒烟进程已返回"

# ---- 3. 闸门 ---------------------------------------------------------------
for _ in $(seq 1 60); do
    grep -q "loss=" "$SMOKE_LOG" 2>/dev/null && break
    grep -q "Traceback" "$SMOKE_LOG" 2>/dev/null && break
    sleep 5
done

if grep -q "Traceback" "$SMOKE_LOG" 2>/dev/null; then
    say "**冒烟失败**，末尾 25 行："
    tail -25 "$SMOKE_LOG" | tee -a "$PIPE_LOG"
    exit 4
fi

if ! grep -q "loss=" "$SMOKE_LOG" 2>/dev/null; then
    say "**冒烟没跑到 loss 行**（既没 Traceback 也没 loss，多半是卡住或被 kill），末尾 25 行："
    tail -25 "$SMOKE_LOG" | tee -a "$PIPE_LOG"
    exit 5
fi
say "冒烟过了，最后一条 loss 行："
grep "loss=" "$SMOKE_LOG" | tail -1 | tee -a "$PIPE_LOG"

# ---- 4. 新增 3 维的逐维 loss ----------------------------------------------
DIAG="$ROOT/data/g1_omnipicker_clear5_d22_100ep/train_diag.jsonl"
if [ -f "$DIAG" ]; then
    say "逐维 loss 自检（第 19/20/21 维必须有限）："
    "$ROOT/.venv/bin/python" - "$DIAG" <<'PY' 2>&1 | tee -a "$PIPE_LOG"
import json, sys, math
# diag 的逐维 loss 是一组 dimNN 键（不是数组），NN 是零填充的两位下标。
last = None
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if line:
        try: last = json.loads(line)
        except json.JSONDecodeError: pass
if last is None:
    print("  **diag 文件是空的** —— 逐维 loss 没写出来，不能判断新维在不在学")
    raise SystemExit(0)
want = [18, 19, 20, 21]          # goal_dist / goal_y / cycle_dist / cycle_idx
names = ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"]
vals = []
for i in want:
    v = last.get(f"dim{i:02d}")
    vals.append(float("nan") if v is None else float(v))
bad = [n for n, v in zip(names, vals) if not math.isfinite(v)]
print(f"  step={last.get('step')}  总 loss={last.get('loss')}")
for i, n, v in zip(want, names, vals):
    print(f"    dim{i:02d} {n:<11} {v:.6f}")
# d19goal 那一档跑到 9990 步时 dim18(goal_dist)=0.0141，拿它当量级参照：
# 新维若高出两三个数量级，多半是归一化尺度坏了；若是 NaN/inf 则直接是错的。
print("  **有非有限值: " + ", ".join(bad) + "**" if bad else "  四维全部有限 ✓")
print(f"  （参照：d19goal 档 9990 步时 goal_dist=0.0141）")
PY
else
    say "  没有 $DIAG —— 跳过逐维自检（不影响是否放正式训练）"
fi

if [ "$NO_TRAIN" = "--no-train" ]; then say "(--no-train) 到此为止"; exit 0; fi

# ---- 5. 正式训练 -----------------------------------------------------------
say "--- 放正式训练 20000 步 ---"
bash "$ROOT/scripts/local/train_d22.sh" >>"$PIPE_LOG" 2>&1
say "正式训练已拉起: PID $(cat /tmp/d22.pid 2>/dev/null)"
say "日志: /home/dgl/lxc/.runlogs/train_d22_100ep.log"
say "===== 流水线完成（训练在后台继续） ====="
