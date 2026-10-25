#!/bin/bash
# 随机化 Clear-All 采集 → g1_omnipicker_clear5_rand，直到 TARGET 集。
# 崩溃/正常退出后若未达目标，sleep 后 --resume 自动重启（可跨 Xid79 等中断）。
#
# 用法: TARGET=100 nohup bash run_clear5rand_bulk.sh > /dev/null 2>&1 &
#   TARGET      目标总集数(默认 100)
#   SETTLE_TOL  闭爪前驻留容差 mm(默认 8；见下方说明)
#   SETTLE_EXTRA 驻留上限步数(默认 60)
#
# ── 2026-09-23 加的两道闸，起因是一次真实的无限空转 ──────────────
# 现象：--resume 撞上一个半初始化的数据集目录(只有 meta/info.json、
#       没有 meta/tasks.jsonl)，LeRobot 读不到 tasks.jsonl 就回退去
#       HuggingFace 拉 repo_id=local/...，本机无外网 ⇒ ConnectTimeout 崩。
#       看门狗只会 sleep 20 再拉起 ⇒ 无限重启、每次都崩在同一处。
# 闸①：起跑前体检，半初始化目录直接拒绝起跑并说明怎么修。
# 闸②：连续 3 次"活不过 180 秒"就放弃，不再无限重启。
set -u
PY=/home/dgl/miniconda3/envs/orcalab/bin/python
WORKDIR=/home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
SCRIPT=$WORKDIR/g1_omnipicker_collect_clear5_lerobot.py
DATA_ROOT=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_rand
DATA_META=$DATA_ROOT/meta/episodes.jsonl
LOG=/tmp/clear5rand_bulk.log
TARGET=${TARGET:-100}
MIN_FREE_GB=20          # 100 集 ≈ 1.2GB，留足余量

# 抓取闸：默认开驻留（j6d 实测组）。
#   tol=8mm 是**不可达**的容差 —— 指令 z 被刻意压到末端可达下限之下，末端稳态
#   永远离目标 22~46mm，所以这道闸从不"通过"，只会退化成固定 60 步驻留。
#   它的真实作用不是"到位才闭爪"，而是"让上一段的暂态衰减完再下压"。
#   实测集良率 22%(无驻留,n=18) → 38%(驻留,n=21)，代价帧数 +1.5%。
SETTLE_TOL=${SETTLE_TOL:-8}
SETTLE_EXTRA=${SETTLE_EXTRA:-60}

cd "$WORKDIR" || exit 1

count_eps() {
    if [ -f "$DATA_META" ]; then grep -c '^' "$DATA_META"; else echo 0; fi
}

# ── 闸①：半初始化目录体检 ──
# 为什么会半初始化：info.json 在**建目录时**就写，tasks.jsonl 要等**首次成功保存**
#   才写(j6d 实测 info 10:18 / tasks 10:42)。所以进程只要在第一集存下来之前崩，
#   就会留下"有 info.json、没有 tasks.jsonl、没有 data/"的壳 —— 这是常规崩溃的
#   产物，不是异常。而 --resume 读到它会 FileNotFoundError，再回退去 HuggingFace
#   拉 repo_id=local/... ⇒ 无外网 ⇒ ConnectTimeout 崩 ⇒ 看门狗无限重启。
# 处理：这种壳**可证明为空**(meta 只有 info.json，没有任何 parquet)，直接删掉重建。
#   若居然存在 parquet 却没有 tasks.jsonl，那才是真损坏 —— 拒绝起跑交给人看。
if [ -f "$DATA_ROOT/meta/info.json" ] && [ ! -f "$DATA_ROOT/meta/tasks.jsonl" ]; then
    if find "$DATA_ROOT" -name '*.parquet' | grep -q .; then
        echo "[watchdog] $(date '+%F %T') 拒绝起跑：$DATA_ROOT 有 parquet 但缺 meta/tasks.jsonl，属真损坏，需人工处理" | tee -a "$LOG"
        exit 1
    fi
    echo "[watchdog] $(date '+%F %T') 发现半初始化空壳(无任何 parquet)，删除后重建：$DATA_ROOT" | tee -a "$LOG"
    rm -rf "$DATA_ROOT"
fi

# ── 起跑前的磁盘闸（ENOSPC 崩过一次，验显存≠验磁盘）──
free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
echo "[watchdog] $(date '+%F %T') 磁盘可用 ${free_gb}G (需 ≥${MIN_FREE_GB}G)" >> "$LOG"
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "[watchdog] $(date '+%F %T') 磁盘不足，拒绝起跑" >> "$LOG"
    exit 1
fi

fast_exits=0
while :; do
    n=$(count_eps)
    echo "[watchdog] $(date '+%F %T') episodes=$n target=$TARGET" >> "$LOG"
    if [ "$n" -ge "$TARGET" ]; then
        echo "[watchdog] $(date '+%F %T') 已达目标 $TARGET 集，退出。" >> "$LOG"
        break
    fi
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "[watchdog] $(date '+%F %T') 磁盘降到 ${free_gb}G，暂停采集" >> "$LOG"
        sleep 600
        continue
    fi

    t0=$(date +%s)
    echo "[watchdog] $(date '+%F %T') 启动采集器 (resume, target=$TARGET, 磁盘 ${free_gb}G)..." >> "$LOG"
    "$PY" "$SCRIPT" \
        --resume \
        --num_episodes "$TARGET" \
        --lerobot_out "$DATA_ROOT" \
        --repo_id "local/g1_omnipicker_clear5_rand" \
        --grasp_dz_mm -12 \
        --settle_tol_mm "$SETTLE_TOL" \
        --settle_max_extra "$SETTLE_EXTRA" \
        --grasp_probe \
        --k_min 1 --k_max 5 \
        --tool_weights 1,10,1,1,1 \
        --plant_mode mixed --plant_all_frac 0.5 \
        --init_jitter_deg 6 \
        >> "$LOG" 2>&1
    rc=$?
    ran=$(( $(date +%s) - t0 ))
    echo "[watchdog] $(date '+%F %T') 采集器退出 rc=$rc，存活 ${ran}s" >> "$LOG"

    # ── 闸②：连续短命退出就别再空转了 ──
    # 注意：**不能靠 rc 判崩溃** —— 采集器把异常 except 掉后是正常退出，rc=0
    # (2026-09-23 实测：AttributeError 崩掉那次打印的是"退出 rc=0，存活 11s")。
    # 只按存活时长判：正常一集要好几分钟，活不过 180s 一定是没跑起来。
    if [ "$ran" -lt 180 ]; then
        fast_exits=$((fast_exits + 1))
        echo "[watchdog] $(date '+%F %T') 短命退出 ${fast_exits}/3" >> "$LOG"
        if [ "$fast_exits" -ge 3 ]; then
            echo "[watchdog] $(date '+%F %T') 连续 3 次活不过 180s，判定为确定性崩溃，停止重试。看 $LOG 末尾的 Traceback。" >> "$LOG"
            exit 1
        fi
    else
        fast_exits=0
    fi

    if ! timeout 3 bash -c 'cat < /dev/null > /dev/tcp/localhost/50051' 2>/dev/null; then
        echo "[watchdog] $(date '+%F %T') sim(50051) 不可达，等待 60s" >> "$LOG"
        sleep 60
    else
        sleep 20
    fi
done
