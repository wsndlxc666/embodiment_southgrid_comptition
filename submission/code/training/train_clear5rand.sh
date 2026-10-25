#!/bin/bash
# clear5_rand 微调：从固定左到右的 43500 基线，热启动到"任意顺序/k 可变"的随机数据集。
#
# 这是**微调**不是重训。理由（详见 config.py 里 pi05_g1_omnipicker_clear5_rand_lora 的注释）：
# 数据只有 132,288 帧，batch4×12000 步 = 48k 样本 = 0.36 epoch，冷启动连"会抓这 5 把"
# 都学不回来；而 43500 已经会全部技能，新数据新增的只是"顺序被打乱"。两边都是 18 维、
# prompt 逐字相同 ⇒ 权重干净加载。
#
# 超参照 d22idxR 那档（同为"从已收敛 ckpt 微调"）：warmup500 / peak1e-4 / floor1e-5。
# 不照老 clear5_lora 的 2e-4+warmup800，那是冷启动档。
#
# 用法: train_clear5rand.sh smoke      # 20 步全链路冒烟
#       train_clear5rand.sh            # 正式，12000 步
#       train_clear5rand.sh 20000      # 指定步数
#       STOP_SERVE=0 train_clear5rand.sh   # 不动策略服务(显存够时)
#       MEM_FRACTION=0.92 train_clear5rand.sh  # 全空卡时预占整块，最快
set -euo pipefail

MODE="${1:-12000}"
CONFIG="pi05_g1_omnipicker_clear5_rand_lora"
EXP="clear5_rand_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_clear5rand.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------------ 显存
# 教训(openpi-train-oom-is-fragmentation)：step0 那个 6.23GiB OOM 是**碎片**不是显存不够，
# GPU 全空时 PREALLOCATE=false 照样崩。正解是**反过来预分配一整块**。
#
# ⚠️ 2026-09-23 实测补一条：**池本身必须大于程序峰值，否则照样 OOM**。
# 我先按"与仿真共存"选了 0.65(池 15974MiB)，崩在 step 0，XLA 自己报出了真实需求：
#     Can't reduce memory use below 4.39GiB ... only reduced to 17.02GiB   ← 峰值 17.02GiB
#     Allocator (GPU_0_bfc) ran out of memory trying to allocate 6.23GiB
# 池 15.6GiB < 峰值 17.02GiB ⇒ 分配器只能报 OOM，跟碎片无关。
#
# 而且"与仿真共存"在这块 3090 上**根本不成立**，算一下就知道：
#     训练峰值 ~17-19GiB + 仿真常驻 5.3GiB = 24.3GiB > 整卡 24GiB
# 所以训练和仿真天然互斥（memory openpi-train-oom-is-fragmentation 里
# "训练期间起不了 OrcaLab 仿真"就是这个意思）。默认独占整块的 0.92。
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.92}"

STOP_SERVE="${STOP_SERVE:-1}"
mkdir -p /home/dgl/lxc/.runlogs

echo "===== 起训前检查 $(date '+%F %T') ====="

# ---- 闸① 语法：坏文件躺到起训才炸过一次(cycle_index.py 文档串首字符被写坏)
for f in src/openpi/training/config.py src/openpi/models/pi0.py src/openpi/models/pi0_config.py; do
    if ! "$PY" -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" 2>&1; then
        echo "✗ 语法没过：$f" >&2; exit 1
    fi
done
echo "✓ 闸① 语法 OK"

# ---- 闸② config 自检 + norm_stats（必须查 create() 的产物；
#      cfg.data.base_config.norm_stats 永远是 None，查它等于没查）
"$PY" - <<'PYEOF'
import pathlib, sys
import numpy as np
import openpi.training.config as C

cfg = C.get_config("pi05_g1_omnipicker_clear5_rand_lora")
print(f"✓ config: {cfg.name}  repo={cfg.data.repo_id}  batch={cfg.batch_size}  steps={cfg.num_train_steps}")
lr = cfg.lr_schedule
print(f"  lr = warmup{lr.warmup_steps} peak{lr.peak_lr} decay{lr.decay_steps} floor{lr.decay_lr}")
print(f"  save/keep = {cfg.save_interval}/{cfg.keep_period}")

fail = []
if lr.peak_lr >= 1e-3:
    fail.append(f"peak_lr={lr.peak_lr} 是冷启动档的量级，微调不该这么高")
if cfg.save_interval == cfg.keep_period:
    fail.append("save_interval == keep_period ⇒ keep_period 的整数倍全被永久保留，等于不删档")

wl = pathlib.Path(cfg.weight_loader.params_path)
if not wl.is_dir():
    fail.append(f"热启动源不存在: {wl}")
else:
    print(f"✓ 热启动源: {wl}")

dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ns = dc.norm_stats
if ns is None:
    fail.append(f"读不到 norm_stats: {cfg.assets_dirs / dc.repo_id}")
else:
    for k in ("state", "actions"):
        q01 = np.asarray(ns[k].q01); q99 = np.asarray(ns[k].q99)
        if q01.shape != (18,) or q99.shape != (18,):
            fail.append(f"{k} 维度不是 18: {q01.shape}")
        if np.any(q99 < q01):
            fail.append(f"{k} 有维度 q99 < q01")
        const = q99 == q01
        if const.any():
            print(f"  {k}: {int(const.sum())} 个常量维(q99==q01) —— 数据里是真常量则无害(分子恒 0)")
    print(f"✓ norm_stats 18 维，q99>=q01 逐维成立")

if fail:
    print("\n✗ 闸② 未通过：")
    for f in fail: print(f"   · {f}")
    sys.exit(1)
print("✓ 闸② 通过")
PYEOF

# ---- 闸③ 显存 / 策略服务
echo "--- 显存现状 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read pid mem; do
    echo "   pid=${pid// /} mem=${mem// /}  $(ps -o args= -p ${pid// /} 2>/dev/null | cut -c1-90)"
done
if [ "$STOP_SERVE" = "1" ]; then
    serve_pids=$(pgrep -f "serve_policy.py" || true)
    if [ -n "$serve_pids" ]; then
        for p in $serve_pids; do
            echo "   停策略服务 pid=$p（用户已授权：训练和评测不能同时）"
            kill "$p" 2>/dev/null || true
        done
        sleep 4
        still=$(pgrep -f "serve_policy.py" || true)
        [ -n "$still" ] && { echo "   ⚠ 策略服务没退干净: $still"; } || echo "   ✓ 策略服务已停"
    else
        echo "   策略服务本来就没在跑"
    fi
fi
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
need_mib=$(python3 -c "print(int(24576*${XLA_PYTHON_CLIENT_MEM_FRACTION}))")
# 预分配是对**整卡**取的，所以要 free 掉的那部分必须已经真的空出来
avail_for_jax=$(( 24576 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) ))
echo "   空卡: free=${free_mib}MiB  预分配要=${need_mib}MiB"
if [ "$need_mib" -gt "$free_mib" ]; then
    echo "✗ 闸③ 未通过：预分配 ${need_mib}MiB > 当前空闲 ${free_mib}MiB"
    echo "   对策：调低 MEM_FRACTION，或 STOP_SERVE=1 停掉策略服务，或关掉仿真"
    exit 1
fi
echo "✓ 闸③ 通过"

# ---- 闸④ 磁盘
# 教训(openpi-ckpt-disk-budget)：单档 8.6G ≈ 17G/千步；keep_period 的整数倍永久保留。
# 2026-09-22 就是只验了显存没验磁盘，ENOSPC 崩掉 5 小时训练。
steps_for_disk=$([ "$MODE" = "smoke" ] && echo 20 || echo "$MODE")
need_gb=$(python3 -c "
steps=$steps_for_disk
import json,math
save,keep=1000,5000
per=8.6
n_saves=steps//save + 1
n_perm=steps//keep          # keep_period 的整数倍永久保留
n_latest=1                  # max_to_keep=1 再留最新一档
# 写档瞬间新旧并存，给 2 倍余量
print(int(math.ceil((n_perm+n_latest)*per*2 + 5)))
")
free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
echo "   需要 ≈${need_gb}G（永久档 + 最新档 + 写档瞬间峰值x2）  可用 ${free_gb}G"
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "✗ 闸④ 未通过：磁盘不足" >&2; exit 1
fi
echo "✓ 闸④ 通过"

echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_rand_smoke"; LOG="/home/dgl/lxc/.runlogs/train_clear5rand_smoke.log"
    echo "===== 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"
    # num_train_steps 与 decay_steps 必须一起传：只改步数不改 decay_steps 的话，
    # 跑过 config 里那个 decay_steps 之后 LR 会一直贴地板
    # (openpi-baseline-lr-floor-from-launcher，就是这么白跑了 42000 步)。
    EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 steps=$TOTAL alloc=BFC预分配${XLA_PYTHON_CLIENT_MEM_FRACTION}" | tee -a "$LOG"

CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留)" | tee -a "$LOG"
fi

nohup "$PY" scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/clear5rand.pid
echo "PID $(cat /tmp/clear5rand.pid)"
echo "日志: $LOG"
