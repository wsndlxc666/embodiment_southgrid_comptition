#!/bin/bash
# idxR 训练：**接 idxP/5000 往下推**，只动两处权重，各自对着一个闭环症状。
#
# 从哪来：用户在 OrcaLab 里跑了 idxP（.runlogs/rollout_idxP_3000.jsonl）拿到反馈 ——
#   {"cycle_idx_end": 2, "placed": {"扳手": true, 其余 false}, "elapsed_total": 165.0}
# 两句用户原话对应两个症状：
#   (1) "知道抓前三个了，但是第四下还是抓的第三个，并没有抓后面的两个"
#   (2) "前几个的抓取效果没有那么好，螺丝刀和电工刀都爪掉了"
#
# **这两个症状是同一个根因**：右爪抓不牢 → 工具中途掉了 → 箱内没有"开爪释放"沿 →
# runner 的 cycle_idx 卡在 2 推不动（cycle_trace 只有 1→2 一次跳变）→ 第 4 下又回到
# 刚做过的那件。所以本轮**先修抓取**，相位权重只是辅助。
# idxP 相对 idxI 的真实进步是**次序对了**：idxI 从第 0 块就往手电筒去（相位/次序错，
# 见 memory openpi-chunk-geometry-1s-horizon 第 4 节），idxP 开始按 扳手→螺丝刀→
# 电工刀(左) 的顺序去抓前三件。别把这两档的失败模式混为一谈。
#
# 相对 idxP 只有**两处**改动（单变量，结果可归因）：
#   [改动1 抓取] loss_weights 右爪 16/17: 5.0 → 8.0（爪权重占比 23.1% → 32.4%）
#   [改动2 相位] cycle_phase_weights ((0,3),(1,2),(4,2)) → 补上 (2,2),(3,2)
#                idxP 实测成分 p0 17.3 | p1 21.2 | p2 14.8 | p3 14.8 | p4 21.2 | p5 10.7
#                p2/p3 是六个里最低的，而模型恰好卡在**进入 p3（手电筒）**。
#                改动后按**绝对样本数**（不是占比%，p0 加权会把总池撑大 1.13x 而稀释
#                所有占比）p2/p3 各 94000 → 134000 = 1.426x，与 p1/p4 齐平；
#                p0/p1/p4/p5 逐样本相同。账见 scripts/local/verify_idxR.py。
#
# 为什么锁定在右爪（norm_stats 实测，不是推测）：整个任务**只用右臂+右爪**完成 ——
#   L爪14/15 q01=q99=0.0480 std=0.0000（100 集全程恒=张开），L臂 std 0.0008~0.0048，
#   只有 R爪16/17 span 0.9518。左臂 0.05 / 左爪 1.0 的低权重是**对的**，别去"修正"它；
#   左爪 loss 低不是因为学得好，是因为它不动（memory openpi-quantile-norm-scale-free）。
#
# 已知代价：p5（收尾段）现在是占比最低的 9.5%。但它的**绝对样本一格没变**（67500），
# 模型现在根本走不到 p5，它不是当前的约束。等 p3/p4 做出来之后下一轮必须补 p5。
#
# 用法: train_d22idxR.sh smoke        # 20 步，全链路冒烟
#       train_d22idxR.sh            # 正式，默认 12000 步
#       train_d22idxR.sh 6000       # 指定步数
set -euo pipefail

MODE="${1:-12000}"
CONFIG="pi05_g1_omnipicker_clear5_d22idxR_lora_12k"
EXP="clear5_d22idxR_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d22idxR.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------- 显存
# step-0 OOM 的真因是**碎片**不是显存不够（GPU 全空也照崩，报同一个 6.23GiB），
# 正解是预分配一整个大块。详见 train_d22idxP.sh 那段注释与 memory
# openpi-train-oom-is-fragmentation。
# 代价：22.6GB 被训练独占，**训练期间起不了 OrcaLab 仿真** —— 别在训的时候开仿真。
if [ "${FORCE_PLATFORM_ALLOC:-0}" = "1" ]; then
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    ALLOC_NOTE="platform（慢，稳态 ~2.6 s/step；峰值低，可与仿真共存）"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
    ALLOC_NOTE="BFC 预分配整块 0.92（快，~1.6 s/step；训练期间仿真起不来）"
fi
mkdir -p /home/dgl/lxc/.runlogs

# ---------------------------------------------------------------- 起训前硬闸
# 教训：2026-09-18 晚 cycle_index.py 文档字符串首字符被写坏，import 就 SyntaxError，
# 没人编译过一次，坏文件一直躺到下一次起训才炸。先编译，编译不过别起。
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
cfg = C.get_config("pi05_g1_omnipicker_clear5_d22idxR_lora_12k")
d = cfg.data.base_config
w = cfg.model.loss_weights
m = sum(w) / len(w)
wl = cfg.weight_loader.params_path
WANT_PHASE = ((0, 3.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0))
WANT_LW = [0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10
print(f"✓ config: {cfg.name}  exp={cfg.exp_name}  steps={cfg.num_train_steps}")
print(f"✓ 相位权重: {d.cycle_phase_weights}")
print(f"✓ 边界过采样: window={d.cycle_boundary_window} weight={d.cycle_boundary_weight}"
      f"  反事实 prob={d.cycle_counterfactual_prob} images={d.cycle_counterfactual_images}")
print(f"✓ 右爪16/17 = {[round(x/m,3) for x in w[16:18]]}（归一化后）"
      f"  左爪14/15 = {[round(x/m,3) for x in w[14:16]]}"
      f"  索引19/20/21 = {[round(x/m,3) for x in w[19:22]]}")
errs = []
if tuple(d.cycle_phase_weights) != WANT_PHASE:
    errs.append(f"相位权重不是设计的 {WANT_PHASE}")
if list(w) != WANT_LW:
    errs.append(f"loss_weights 不是设计的（长度 {len(w)}，右爪 {w[16:18]}）")
if not pathlib.Path(wl).is_dir():
    errs.append(f"热启动源不存在: {wl}")
if "5000" not in str(wl):
    errs.append(f"热启动源不是 idxP/5000（本轮是接 idxP 往下推）: {wl}")
for e in errs:
    print(f"✗ {e}")
print(f"✓ 热启动源存在: {wl}")
sys.exit(1 if errs else 0)
PYEOF
echo "  （完整的成分/单变量对账见 scripts/local/verify_idxR.py）"
echo "GPU 现状: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)  分配器: $ALLOC_NOTE"
echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_d22idxR_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22idxR_smoke.log"
    echo "===== d22idxR 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== d22idxR 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
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
echo $! > /tmp/d22idxR.pid
echo "PID $(cat /tmp/d22idxR.pid)"
echo "日志: $LOG"
echo "盯这行: tail -f $LOG"
