#!/bin/bash
# Stage 2「目标位置」oracle 单变量对照 —— 2026-09-22 重启设计。
#
# 与基线 pi05_g1_omnipicker_clear5_lora **只差一处**：state 多 3 维
# (dims 18..20 = tgt_rel = 目标 − ee)。数据集同为 181 集、超参逐字相同、
# 热启动同为 pi05_base。所以两档的任何差异都可归因到「模型能不能用上目标位置」。
#
# 这三 3 维是 **oracle**：target 取每集实测抓取点 = 完美感知系统的输出。
# 工具静止、第 0 帧即可观测，故非时间泄漏。结论单边：
#   oracle 无提升 -> 任何 vision/depth 只会更差，depth 路线整体可砍
#   oracle 有提升 -> 才值得投入 perception
#
# ---------------------------------------------------------------- 显存策略
# 两次失败的都是"分配器/pool 太小"，不是"卡上没地方"：
#  · ALLOC=platform 且策略服务还占着 8.5GB 时，step0 OOM：
#    `RESOURCE_EXHAUSTED: Failed to allocate request for 136.69MiB`（当时只剩 9.6GB）。
#  · BFC MEM_FRACTION=0.70 预占 17.2GB，**pool 本身**不够：报
#    `Out of memory while trying to allocate 6692584152 bytes`（需要再多 6.69GB），
#    同时 XLA 说 `Can't reduce memory use below 5.51GiB by remat`。
#
# 结论：这个模型实测要 **~17.7GB**。BFC 0.70=17.2GB 差一点点，而 0.92=22.6GB
# 在仿真占 5.3GB 时又放不下。**所以默认走 platform 按需分配**（已实测可跑通 20 步
# 全链路，2~4.5 s/it），它会自然长到 17.7GB 而不是一次性预占。
#   代价：比满卡 BFC 慢（memory: 碎片型 OOM 的教训仍在，platform 峰值低所以不触发）。
#   收益：**仿真可以继续开着**，只在空闲不够时才需要腾卡。
#
# ⚠ 已被停掉的：策略服务 `serve_policy.py --port 8010`（占 8.5GB，用户已授权可关）。
#   恢复脚本: /home/dgl/lxc/.runlogs/restart_serve_policy.sh
#   仿真 `orcalab.main`（5.3GB）**不动** —— 用户开着它。
#
# 用法: train_stage2.sh smoke     # 20 步全链路冒烟，先跑这个
#       train_stage2.sh           # 正式，默认 12000 步
#       train_stage2.sh 6000      # 指定步数
set -euo pipefail

MODE="${1:-12000}"
CONFIG="pi05_g1_omnipicker_clear5_stage2_lora"
EXP="clear5_stage2_oracle_181ep"
LOG="/home/dgl/lxc/.runlogs/train_stage2.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 分配器可用环境变量覆盖，默认 platform（实测唯一跑通的）。
#   ALLOC=platform  按需申请；实测长到 ~17.7GB，与仿真共存可跑通（默认）
#   ALLOC=bfc       BFC 预占。BFC_MEM_FRACTION 对**整卡** 24576MiB 取，不是对空闲量。
#                   ⚠ 0.70=17.2GB 实测 pool 不够（差一点点）；要 0.92 才够，
#                   但那需要整卡基本空着 —— 即训练期间仿真起不来。
ALLOC="${ALLOC:-platform}"
BFC_MEM_FRACTION="${BFC_MEM_FRACTION:-0.70}"
if [ "$ALLOC" = "platform" ]; then
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION
    # 实测：冒烟时进程峰值 17720 MiB，起始空闲 18165 MiB（余量仅 ~445 MiB）——
    # 紧，但 20 步全链路跑通、零报错。所以阈值定在实测值略下方，不是拍脑袋的 1.5GB 余量。
    NEED_MIB="${PLATFORM_NEED_MIB:-18000}"
    ALLOC_DESC="platform(按需, 实测峰值~17.7GB)"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$BFC_MEM_FRACTION
    NEED_MIB=$(python3 -c "print(int(24576*$BFC_MEM_FRACTION))")
    ALLOC_DESC="bfc(预占${BFC_MEM_FRACTION})"
fi
mkdir -p /home/dgl/lxc/.runlogs

# ---------------------------------------------------------------- 起训前硬闸
# 教训（memory）：2026-09-18 有个文件首字符被写坏、import 就 SyntaxError，
# 没人编译过一次，坏文件一直躺到下次起训才炸。先编译，编译不过别起。
echo "===== 起训前检查 $(date '+%F %T') ====="
for f in src/openpi/training/config.py src/openpi/training/data_loader.py \
         src/openpi/policies/g1_omnipicker_policy.py; do
    if ! "$PY" -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" 2>&1; then
        echo "✗ 语法检查没过：$f" >&2; exit 1
    fi
done
echo "✓ 相关文件语法 OK"

"$PY" - <<'PYEOF'
import json, pathlib, sys
import openpi.training.config as C

cfg = C.get_config("pi05_g1_omnipicker_clear5_stage2_lora")
errs = []

# config 自洽
if cfg.data.repo_id != "g1_omnipicker_clear5_stage2_181ep":
    errs.append(f"repo_id 不对: {cfg.data.repo_id}")
if cfg.data.action_dim != 18:
    errs.append(f"action_dim 应为 18（action 未改）: {cfg.data.action_dim}")
if cfg.model.action_dim != 32:
    errs.append(f"模型 action_dim 应为 32: {cfg.model.action_dim}")

# LR 日程必须与基线实际跑的那条逐字相同，否则多一个变量、对照作废。
# 基线 = _s1500 被 train_local_3090.sh 启动且**未在命令行覆盖** lr_schedule。
_base = C.get_config("pi05_g1_omnipicker_clear5_lora_s1500").lr_schedule
_lr_keys = ("warmup_steps", "peak_lr", "decay_steps", "decay_lr")
for k in _lr_keys:
    mine, theirs = getattr(cfg.lr_schedule, k, None), getattr(_base, k, None)
    if mine != theirs:
        errs.append(f"lr_schedule.{k} = {mine}，基线是 {theirs}（单变量被破坏）")

# 数据集自洽：state 21 / action 18
info = json.load(open(f"data/{cfg.data.repo_id}/meta/info.json"))
s = info["features"]["observation.state"]["shape"][0]
a = info["features"]["action"]["shape"][0]
if s != 21: errs.append(f"state 应为 21 维，实际 {s}")
if a != 18: errs.append(f"action 应为 18 维，实际 {a}")
names = info["features"]["observation.state"]["names"][0]
if names[-3:] != ["tgt_rel_x", "tgt_rel_y", "tgt_rel_z"]:
    errs.append(f"末尾 3 维名不对: {names[-3:]}")

# 数据量与基线一致
base = json.load(open("data/g1_omnipicker_clear5_train/meta/info.json"))
if info["total_episodes"] != base["total_episodes"]:
    errs.append(f"集数 {info['total_episodes']} != 基线 {base['total_episodes']}（单变量被破坏）")
if info["total_frames"] != base["total_frames"]:
    errs.append(f"帧数 {info['total_frames']} != 基线 {base['total_frames']}")

# norm_stats 必须已经算好（用 config 解析出的真实 assets_dirs，
# 别查 cfg.data.assets.assets_dir —— 那里是 None，Path(None) 直接 TypeError）
ns = pathlib.Path(cfg.assets_dirs) / cfg.data.repo_id / "norm_stats.json"
if not ns.is_file():
    errs.append(f"norm_stats 不存在: {ns}（先跑 norm_stats_direct.py）")
else:
    d = json.load(open(ns))["norm_stats"]          # 多一层 "norm_stats"
    if len(d["state"]["mean"]) != 21:
        errs.append(f"norm_stats state 长度 {len(d['state']['mean'])} != 21")
    if len(d["actions"]["mean"]) != 18:
        errs.append(f"norm_stats actions 长度 {len(d['actions']['mean'])} != 18")
    if len(d["state"]["q01"]) != 21 or len(d["state"]["q99"]) != 21:
        errs.append("norm_stats 缺 q01/q99 —— pi05 走分位数归一化，缺了会 assert 挂")

print(f"✓ config  : {cfg.name}  exp={cfg.exp_name}")
print(f"✓ 数据    : {cfg.data.repo_id}  {info['total_episodes']} 集 / {info['total_frames']} 帧"
      f"  state {s} / action {a}")
print(f"✓ 热启动  : {cfg.weight_loader.params_path}")
print(f"✓ norm    : {ns}")
for e in errs: print(f"✗ {e}")
sys.exit(1 if errs else 0)
PYEOF
# 显存前置闸：BFC 要一次性预占 NEED_MIB，空闲不够就别起，免得白编译一遍再 OOM。
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "  需要 ${NEED_MIB} MiB / 当前空闲 ${FREE_MIB} MiB   分配器=${ALLOC_DESC}"
if [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
    echo "✗ 空闲显存不够预占。占用者：" >&2
    for p in $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | cut -d, -f1); do
        printf "    PID %-8s %s MiB  %s\n" "$p" \
            "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$p," | cut -d, -f2 | tr -d ' MiB')" \
            "$(ps -p $p -o args= 2>/dev/null | cut -c1-70)" >&2
    done
    echo "  策略服务可用 /home/dgl/lxc/.runlogs/restart_serve_policy.sh 事后恢复。" >&2
    exit 1
fi
echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_stage2_smoke"; LOG="/home/dgl/lxc/.runlogs/train_stage2_smoke.log"
    echo "===== 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"
    # ⚠ 这里**只**覆盖 num_train_steps，绝不覆盖 lr_schedule.decay_steps。
    # 基线的 LR 在 1500 步就贴到 2e-5 地板并一直留在那儿（见 config 里的注释）；
    # 传了 decay_steps=$TOTAL 就等于"顺便把超参也改了"，Stage 2 不再是单变量。
    EXTRA=(--num_train_steps "$TOTAL")
    echo "===== 正式起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP total_steps=$TOTAL mode=$MODE alloc=$ALLOC_DESC" | tee -a "$LOG"

CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录" | tee -a "$LOG"
fi

# ---------------------------------------------------------------- 磁盘闸
# 教训（2026-09-22 那晚 5 小时训练全丢）：起训前我只验了显存，**没验磁盘**。
# 崩在写第 16 个存档（step 8000）时：
#   ValueError: RESOURCE_EXHAUSTED ... [OS error 28: ENOSPC No space left on device]
# 每个存档实测 8.6G，12000 步 = 24 个 = 206G，而当时根分区只剩 ~135G。
# **这是起训前就能算出来的必然，不是意外。** 所以补一道跟显存闸对称的磁盘闸。
CKPT_GIB="${CKPT_GIB:-9}"      # 实测单存档 ~8.6G，取 9 保守
KEEP_PERIOD=$("$PY" -c "import openpi.training.config as C; print(C.get_config('$CONFIG').keep_period)")
if [ "$MODE" = "smoke" ]; then
    KEEP_N=1                    # smoke 带 --overwrite，不累积
else
    # checkpoints.py 是 max_to_keep=1 + keep_period=K：永久留存的只有 K 的整数倍，
    # 外加最新那个。所以留存数 = steps/K + 1，不是 steps/save_interval。
    KEEP_N=$(( TOTAL / KEEP_PERIOD + 1 ))
fi
NEED_GIB=$(python3 -c "print(int($KEEP_N*$CKPT_GIB+12))")   # +12G 余量给 orbax 临时目录/日志
FREE_GIB=$(df -BG --output=avail /home/dgl/lxc | tail -1 | tr -dc '0-9')
echo "  磁盘：留存 ${KEEP_N} 个存档(keep_period=${KEEP_PERIOD}) × ${CKPT_GIB}G" \
     "= 需要 ~${NEED_GIB}G / 当前空闲 ${FREE_GIB}G" | tee -a "$LOG"
if [ "$FREE_GIB" -lt "$NEED_GIB" ]; then
    {
        echo "✗ 磁盘不够（这正是 2026-09-22 崩掉的原因）。当前大户："
        du -sh /home/dgl/lxc/openpi/checkpoints/* 2>/dev/null | sort -rh | head -6
        echo "  可清的非本项目项：/tmp/eval_g1_tool_lerobot_stream（历次评测残留，26G）、"
        echo "  /var/log/journal（journalctl --vacuum-size=1G）。"
    } >&2
    exit 1
fi

# num_train_steps 单独传：LR 日程留在 config 里不动（见上面的警告）。
nohup "$PY" scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/stage2.pid
echo "PID $(cat /tmp/stage2.pid)"
echo "日志: $LOG"
echo "盯: tail -f $LOG"
