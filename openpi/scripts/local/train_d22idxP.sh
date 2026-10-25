#!/bin/bash
# idxP 训练：**按相位加权采样**，把 pilot 边界过采样无意中饿死的相位 0（扳手，必须第一个
# 做的那件）补回来。热启动源与 idxI **相同**（idxI/14000）⇒ idxI 就是本轮的对照组，
# 唯一变量是 cycle_phase_weights。
#
# 为什么要补相位 0：cycle_boundary_window=60 / weight=10 的边界窗口只落在相位 1..5 的
# 开头（帧 0 那段 home 位姿被 include_episode_start=False 排除），于是每轮成分是
#     p0 7.6% | p1..p4 各 19.6% | p5 14.1%
# —— "必须第一个抓的那件工具"样本最少，赤字 2.6x，而这正好是 idxI 的表现：它把第 3、4
# 件（电工刀(左)、手电筒）完整做对，却整轮没碰过前两件。见 memory
# openpi-chunk-geometry-1s-horizon 第 5 节。
#
# 权重选 ((0,3.0),(1,2.0),(4,2.0)) 的代价：p2/p3（当前唯一做对的两件）占比从
# 19.6% 降到 14.8%。这是拿已经会的换还不会的 —— 故意如此，见 verify_phase_weight.py 3b。
#
# 用法: train_d22idxP.sh smoke        # 20 步，全链路冒烟
#       train_d22idxP.sh            # 正式，默认 20000 步
#       train_d22idxP.sh 8000       # 指定步数
set -euo pipefail

MODE="${1:-20000}"
CONFIG="pi05_g1_omnipicker_clear5_d22idxP_lora_20k"
EXP="clear5_d22idxP_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d22idxP.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------- 显存：这次真的查清了
# idxI 那版脚本注释把 step-0 OOM 归因于"OrcaLab 仿真常驻占显存"，并据此写死
# XLA_PYTHON_CLIENT_ALLOCATOR=platform。**2026-09-19 实测证明这个归因是错的**：
# 仿真已关、GPU 全空（392 MiB）时，PREALLOCATE=false 的 BFC 走法**照样**在 step 0 崩，
# 报的是同一个 "ran out of memory trying to allocate 6.23GiB (6692584192 bytes)"。
# 崩点现场 GPU 是 17071/24576 MiB —— 总量还剩 7.5GB，够这块 6.23GiB，却还是失败。
# ⇒ 真正的死因是**碎片**：BFC 池子靠倍增长大，要再扩就得翻倍到 34GB（超卡），
#    于是宁可报 OOM 也不给这 6.23GiB 找一块连续空间。跟仿真在不在没关系。
# 正解不是"关掉预分配"，恰恰相反 —— **预分配一整个大块**才没有碎片：
#   PREALLOCATE=true + MEM_FRACTION=0.92 → 冒烟 20/20 通过并存档（exit=0）。
# 代价：这 22.6GB 被训练独占，**训练期间起不了 OrcaLab 仿真**（所以别在训的时候开仿真）。
# 想要慢而"随用随要"的老路子（峰值低、能和仿真共存）: FORCE_PLATFORM_ALLOC=1 train_d22idxP.sh
if [ "${FORCE_PLATFORM_ALLOC:-0}" = "1" ]; then
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    ALLOC_NOTE="platform（慢，稳态 ~2.6 s/step；峰值低，可与仿真共存）"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
    ALLOC_NOTE="BFC 预分配整块 0.92（快，无碎片；训练期间仿真起不来）"
fi
mkdir -p /home/dgl/lxc/.runlogs

# ---------------------------------------------------------------- 起训前硬闸
# 教训：2026-09-18 晚 cycle_index.py 的文档字符串首字符被写坏，import 就 SyntaxError，
# 而没人编译过一次 —— 坏文件一直躺到下一次起训才炸。先编译，编译不过别起。
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
cfg = C.get_config("pi05_g1_omnipicker_clear5_d22idxP_lora_20k")
d = cfg.data.base_config
w = cfg.model.loss_weights
m = sum(w) / len(w)
wl = cfg.weight_loader.params_path
WANT = ((0, 3.0), (1, 2.0), (4, 2.0))
print(f"✓ config: {cfg.name}  exp={cfg.exp_name}  steps={cfg.num_train_steps}")
print(f"✓ 相位权重: {d.cycle_phase_weights}")
print(f"✓ 边界过采样: window={d.cycle_boundary_window} weight={d.cycle_boundary_weight}"
      f"  反事实 prob={d.cycle_counterfactual_prob} images={d.cycle_counterfactual_images}")
print(f"✓ 索引权重(归一化后) dim19/20/21 = {[round(x/m,3) for x in w[19:22]]}"
      f"  右爪16/17 = {[round(x/m,3) for x in w[16:18]]}")
if tuple(d.cycle_phase_weights) != WANT:
    print(f"✗ 相位权重不是设计的 {WANT} —— 别起训"); sys.exit(1)
if not pathlib.Path(wl).is_dir():
    print(f"✗ 热启动源不存在: {wl}"); sys.exit(1)
print(f"✓ 热启动源存在: {wl}")
PYEOF
echo "GPU 现状: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)  分配器: $ALLOC_NOTE"
echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_d22idxP_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22idxP_smoke.log"
    echo "===== d22idxP 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== d22idxP 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE alloc=$ALLOC_NOTE" | tee -a "$LOG"

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
echo $! > /tmp/d22idxP.pid
echo "PID $(cat /tmp/d22idxP.pid)"
echo "日志: $LOG"
echo "盯这行: tail -f $LOG"
