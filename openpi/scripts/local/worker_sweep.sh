#!/bin/bash
# 扫 num_workers 稳定性/速度。用法: bash worker_sweep.sh
cd /home/k24lxc/openpi
PY=/home/k24lxc/openpi/.venv/bin/python
for W in 1 2 4; do
  echo "===== sweep W=$W batch=12 n=6 ====="
  timeout 240 $PY /home/k24lxc/probe_dataloader.py $W 12 6 > /home/k24lxc/sweep_w$W.log 2>&1
  echo "exit=$?"
  grep "^\[probe\]" /home/k24lxc/sweep_w$W.log
  Z=$(ps -eo stat,comm | grep python | grep -c '^Z' || true)
  echo "zombie_python=$Z"
done
