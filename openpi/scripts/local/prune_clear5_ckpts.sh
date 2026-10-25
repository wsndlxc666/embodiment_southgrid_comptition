#!/bin/bash
# clear5 ckpt 瘦身:每 500 步照常存档(瞬时记录),但只保留
#   1) 步数为 10000 整数倍的"参考档"(10000/20000/30000/...)
#   2) 最新完整档(供续训 resume / 即时评测)
# 其余已完成档删除。安全:仅删含 _CHECKPOINT_METADATA 的完整档、且从不删最新。
# 用法: prune_clear5_ckpts.sh
set -uo pipefail
D=/home/dgl/lxc/openpi/checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/clear5_v1_l3090_lora
[ -d "$D" ] || { echo "目录不存在: $D"; exit 1; }

# 最新完整档 = 含 metadata 的数字档里最大的
LATEST=""
for s in $(ls "$D" | grep -E '^[0-9]+$' | sort -n); do
  [ -f "$D/$s/_CHECKPOINT_METADATA" ] && LATEST="$s"
done
if [ -z "$LATEST" ]; then echo "无完整档,跳过"; exit 0; fi

DEL=()
for s in $(ls "$D" | grep -E '^[0-9]+$' | sort -n); do
  [ "$s" = "$LATEST" ] && continue          # 最新档永不留删
  [ $((s % 10000)) -eq 0 ] && continue       # 10000 倍数参考档保留
  [ -f "$D/$s/_CHECKPOINT_METADATA" ] && DEL+=("$s")
done

if [ ${#DEL[@]} -eq 0 ]; then
  echo "无需删除。保留: 参考=$(ls "$D" | grep -E '^[0-9]+$' | awk '$1%10000==0' | sort -n | tr '\n' ' ') 最新=$LATEST"
  exit 0
fi

echo "删除 ${#DEL[@]} 档: ${DEL[*]}"
for s in "${DEL[@]}"; do
  rm -rf "$D/$s"
done
echo "保留: 参考=$(ls "$D" | grep -E '^[0-9]+$' | awk '$1%10000==0' | sort -n | tr '\n' ' ') 最新=$LATEST"
du -sh "$D"
