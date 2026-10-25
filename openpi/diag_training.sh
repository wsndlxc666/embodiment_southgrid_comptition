#!/usr/bin/env bash
# =====================================================================
# GPU 故障定位脚本 —— "到底哪里出问题", 全部写进日志
#
# 用法:   bash diag_training.sh [步数]      (默认 120 步 ≈ 8 分钟)
# 产出:   diag_<时间戳>.log                  带时间戳的完整日志
#         gpu_monitor_<时间戳>.csv            每 2 秒一次 GPU 温度/功耗/利用率采样
#
# 崩溃定位判断依据:
#   - 日志停在 "[Phase] 初始化数据加载器"           → 问题在数据读取
#   - 日志停在 "Initialized train state"           → 问题在权重加载/模型初始化
#   - 日志停在 "[Phase] 训练状态就绪"之后无 Step    → 问题在 JIT 编译(首次 kernel)
#   - 日志有 "Step N" 但后续不再增加, 且 csv 里温度/功耗异常攀升 → 过热/供电
#   - 日志报 "无法访问 GPU / No devices were found"  → 驱动/硬件锁死, 需重启+查硬件
#   - 日志正常走完 SMOKE_STEPS 步                  → 代码与配置无问题, 是偶发驱动/硬件
# =====================================================================
set -u

OPENPI=/home/dgl/lxc/openpi
cd "$OPENPI" || { echo "找不到 $OPENPI"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG="$OPENPI/diag_${TS}.log"
CSV="$OPENPI/gpu_monitor_${TS}.csv"
SMOKE_STEPS="${1:-120}"

log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

log "================ 诊断开始 ================"
log "主机=$(hostname)  时间=$(date)  运行时长=$(uptime -p)"

# ---- Phase 0: 基线 ----
log "[P0] GPU 基线诊断"
nvidia-smi --query-gpu=index,name,driver_version,power.limit,temperature.gpu,memory.total --format=csv,noheader | tee -a "$LOG"
GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l)
log "[P0] 检测到 GPU 数量 = ${GPU_N}"

# ---- Phase 1: 后台 GPU 监控(每 2 秒采样一次) ----
log "[P1] 启动 GPU 监控 -> $CSV"
echo "ts_unix,temp_c,power_w,util_pct,mem_mib,clocks_mhz,fan_pct" > "$CSV"
(
  while true; do
    line=$(timeout 4 nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm,fan.speed --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$line" ]; then
      echo "$(date +%s),$(echo "$line" | tr -d ' ' | tr ',' ',')" >> "$CSV"
    else
      echo "$(date +%s),N/A,N/A,N/A,N/A,N/A,N/A" >> "$CSV"
    fi
    sleep 2
  done
) &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null' EXIT
log "[P1] 监控进程 PID=$MON_PID"

# ---- Phase 2: 冒烟训练(覆盖 num_train_steps/log_interval) ----
log "[P2] 启动冒烟训练 ${SMOKE_STEPS} 步(正常约需 $((SMOKE_STEPS * 4 / 60)) 分钟)..."
CUDA_VISIBLE_DEVICES=0 \
HF_LEROBOT_HOME="$OPENPI/data" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=offline \
uv run scripts/train.py pi05_g1_omnipicker_tool_lora \
  --exp-name="diag_${TS}" --overwrite \
  --num_train_steps="$SMOKE_STEPS" \
  --save_interval=100 \
  --log_interval=5 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
log "[P2] 训练进程退出码 = $RC"

# ---- Phase 3: 事后检查 ----
kill "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null
log "[P3] 训练结束后 GPU 状态"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader | tee -a "$LOG" || log "[P3] 训练结束后 GPU 已不可访问(锁死)"
log "[P3] 提示: 若想看内核 XID 错误, 重启后运行:  sudo dmesg | grep -iE 'nvidia|xid'"

# ---- 汇总 ----
log "==== 汇总 ===="
if [ "$RC" -eq 0 ]; then
  log "结果: 训练正常完成 ${SMOKE_STEPS} 步, 本轮未复现 GPU 崩溃(代码/配置无问题)"
else
  log "结果: 训练失败, 退出码=$RC"
  log "崩溃前 GPU 采样(最后 20 行, 时间戳为 epoch 秒):"
  tail -20 "$CSV" | tee -a "$LOG"
  log "从上表看崩溃前温度/功耗趋势即可判断是否过热或供电不足"
fi
log "完整日志: $LOG"
log "GPU 监控: $CSV"
