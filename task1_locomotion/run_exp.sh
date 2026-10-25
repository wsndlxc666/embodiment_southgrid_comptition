#!/usr/bin/env bash
# orcalab 实验启动器
# 用 orcalab 环境的 Python 绝对路径直接运行，绕开 `conda run` 在交互终端里的
# PYTHONPATH/PYTHONHOME 污染问题（表现：No module named 'orca_gym'/'orca_scorer_client'）。
# 自动把终端输出存到 exp_logs/ 下带时间戳的日志文件。
#
# 用法:
#   ./run_exp.sh <script.py> [参数...]
#
#   示例:
#   ./run_exp.sh run_task1_heft_navigation.py --no-score
#   ./run_exp.sh run_task1_failure_tests.py --case no-b --team-id xxx --team-token yyy
#
# 如需换 Python（例如 Docker/其他机器），设置环境变量 ORCA_PY。
set -euo pipefail

PY="${ORCA_PY:-/home/dgl/miniconda3/envs/orcalab/bin/python}"
SCRIPT="${1:?用法: ./run_exp.sh <script.py> [参数...]}"
shift

if ! command -v "$PY" >/dev/null 2>&1; then
    echo ">> 找不到 Python: $PY（可用环境变量 ORCA_PY 指定）" >&2
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p exp_logs
STAMP=$(date +%m%d_%H%M%S)
BASE=$(basename "$SCRIPT" .py)
LOG="exp_logs/${BASE}_${STAMP}.log"

echo ">> python:  $PY"
echo ">> script:  $SCRIPT $*"
echo ">> log:     $LOG"
echo "========================================"

# tee 同时输出到终端和日志；tee 模式下 Python 输出被行缓冲，
# 用 python -u 保证实时可见。
"$PY" -u "$SCRIPT" "$@" 2>&1 | tee "$LOG"
