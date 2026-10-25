#!/bin/bash
# 改法 A 训练：反事实样本**连图像一起换成异相位供体帧**，让 cycle_idx 成为相位身份的唯一解释。
#
# 与 train_d22idx.sh（pilot3k）的区别，只有两处（见 config.py 里那段注释）：
#   1. cycle_counterfactual_images=True   ← 本轮主角，凭空造出那条为 0 的 idx 梯度
#   2. loss_weights 的 dims19/20/21 由 1.0 → 3.0   ← 用户要的"加大索引权重"
# 热启动源与 pilot **相同**（d22 自己的 19000）⇒ pilot 就是本轮的对照组。
#
# 为什么必须换图：pilot 只换 dims19/20、图像不动，训完模型在 index 通道内部把
# (19,20)↔21 的矛盾解开了（dim19 的 loss 掉 191 倍），**但手臂没跟着 idx 走** ——
# 因为"读图"和"读 idx"给出同一个 action，指向 idx 的相对梯度恒为 0。
# 所以 2999 步的结论不是"欠拟合"，是"这个信号不存在"。
#
# 用法: train_d22idxI.sh smoke        # 20 步，全链路冒烟
#       train_d22idxI.sh            # 正式，默认 20000 步
#       train_d22idxI.sh 8000       # 指定步数
set -euo pipefail

MODE="${1:-20000}"
CONFIG="pi05_g1_omnipicker_clear5_d22idxI_lora_20k"
EXP="clear5_d22idxI_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d22idxI.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
# ⚠️ 2026-09-19 更正：下面这段归因**是错的**，别照抄。真因不是"仿真占着显存"，而是 BFC
# 碎片 —— 仿真已关、GPU 全空（392MiB）时 PREALLOCATE=false 照样在 step 0 崩同一个 6.23GiB，
# 崩点现场还有 7.5GB 空闲。**正解是反过来预分配一整块**：PREALLOCATE=true +
# MEM_FRACTION=0.92（冒烟 20/20 通过），比 platform 快。见 scripts/local/train_d22idxP.sh
# 的注释块，或 memory openpi-train-oom-is-fragmentation。本文件保留原样作为当时的记录。
#
# 显存：OrcaLab 仿真常驻约 5.3GB，训练本身要 10.7GiB 起。实测只压 MEM_FRACTION **没用** ——
# 它只管**预分配**那一下，BFC 分配器之后照样贪心增长，2026-09-18 23:40 冒烟就在 step 0
# 报 "ran out of memory trying to allocate 6.23GiB"（GPU 当时已占 23281/24576 MiB）。
# 必须**关掉预分配**改成按需分配，让 JAX 用剩下的那 ~19GB。
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 但**光这样还不够**：BFC 分配器不预分配之后照样贪心增长，实测仍会要一整块 6.23GiB
# 连续内存而失败。换成 platform 分配器（每次 cudaMalloc、不做缓存）才真的把峰值压下来。
# 代价是慢：稳态 2.6 s/step（BFC + 无仿真时是 1.78），20k 步约 14.4 小时。
# 冒烟实测：20/20 步跑通并存档。
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
mkdir -p /home/dgl/lxc/.runlogs

# ---------------------------------------------------------------- 起训前硬闸
# 教训：2026-09-18 晚 cycle_index.py 的文档字符串首字符被写坏（三个引号的头一个变成
# 全角逗号），整个模块 import 就 SyntaxError。因为没人编译过一次，这个坏文件一直躺到
# 下一次起训才会炸 —— 而那时你可能已经等了几小时。所以这里先编译，编译不过就别起。
echo "===== 起训前检查 $(date '+%F %T') ====="
for f in src/openpi/training/cycle_index.py src/openpi/training/config.py \
         src/openpi/training/data_loader.py; do
    if ! "$PY" -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" 2>&1; then
        echo "✗ 语法检查没过：$f —— 先修好再起训" >&2
        exit 1
    fi
done
echo "✓ 三个被改过的文件语法都 OK"

"$PY" - <<'PYEOF'
import pathlib, sys
import openpi.training.config as C
cfg = C.get_config("pi05_g1_omnipicker_clear5_d22idxI_lora_20k")
d = cfg.data.base_config
w = cfg.model.loss_weights
m = sum(w) / len(w)
wl = cfg.weight_loader.params_path
print(f"✓ config: {cfg.name}  exp={cfg.exp_name}  steps={cfg.num_train_steps}")
print(f"✓ 反事实: window={d.cycle_boundary_window} weight={d.cycle_boundary_weight} "
      f"prob={d.cycle_counterfactual_prob} images={d.cycle_counterfactual_images}")
print(f"✓ 索引权重(归一化后) dim19/20/21 = {[round(x/m,3) for x in w[19:22]]}"
      f"  右爪16/17 = {[round(x/m,3) for x in w[16:18]]}")
if not d.cycle_counterfactual_images:
    print("✗ 换图没开 —— 这轮就等于白跑（pilot 已证只换 dim 无效）"); sys.exit(1)
if not pathlib.Path(wl).is_dir():
    print(f"✗ 热启动源不存在: {wl}"); sys.exit(1)
print(f"✓ 热启动源存在: {wl}")
PYEOF
echo "GPU 现状: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_d22idxI_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22idxI_smoke.log"
    echo "===== d22idxI 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== d22idxI 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE" | tee -a "$LOG"

CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留)" | tee -a "$LOG"
fi

# 必须把 num_train_steps 和 decay_steps 一起传：只改步数不改 decay_steps 的话，
# 跑超过 config 里那个 decay_steps 之后 LR 会一直贴在地板上。
nohup "$PY" scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d22idxI.pid
echo "PID $(cat /tmp/d22idxI.pid)"
echo "日志: $LOG"
echo "盯这行: tail -f $LOG"
