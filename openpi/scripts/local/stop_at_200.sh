#!/bin/bash
# 落盘总数到 200 即停采(watchdog+collector),供本机训练腾 GPU。每 20s 轮询。
META=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5/meta/episodes.jsonl
WATCHDOG=606778   # run_clear5_to1000.sh (bash)
COLLECTOR=606784  # g1_omnipicker_collect_clear5_lerobot.py (python)
LOG=/tmp/clear5_stop_at_200.log
echo "[stop_at_200] $(date '+%F %T') 开始轮询,目标 total>=200" > "$LOG"
while :; do
  n=$(grep -c '^' "$META" 2>/dev/null || echo 0)
  if [ "$n" -ge 200 ]; then
    echo "[stop_at_200] $(date '+%F %T') 已达 $n 集,停 watchdog(606778) 再停 collector" >> "$LOG"
    kill "$WATCHDOG" 2>/dev/null && echo "watchdog TERM 已发" >> "$LOG"
    sleep 1
    kill -INT "$COLLECTOR" 2>/dev/null && echo "collector SIGINT 已发(优雅收尾当前数据单元)" >> "$LOG"
    for i in 1 2 3 4 5 6; do
      sleep 5
      if ! kill -0 "$COLLECTOR" 2>/dev/null; then echo "collector 已退出" >> "$LOG"; break; fi
    done
    if kill -0 "$COLLECTOR" 2>/dev/null; then kill -TERM "$COLLECTOR"; echo "collector 升级 TERM" >> "$LOG"; fi
    echo "[stop_at_200] $(date '+%F %T') DONE n=$n" >> "$LOG"
    break
  fi
  sleep 20
done
