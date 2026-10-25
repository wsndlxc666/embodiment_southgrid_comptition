#!/bin/bash
# 把训练过程中每 1000 步的存档**硬链接**留一份到 _keep/。
#
# 为什么需要：CheckpointManager 的选项是 max_to_keep=1 + keep_period=5000，
# 也就是**每存一个新档就把上一档删掉**，只有 5000 的整数倍永久保留。所以 1000/2000/…
# 这些滚动档只活 ~28 分钟就没了 —— 想在中途取一个存档来评测，窗口极窄
# （2026-09-19 实测：14:39 存好 1000，15:05 就被 2000 挤掉）。
#
# 为什么用硬链接而不是 cp：cp 一份 8.7G，每 28 分钟一次、20 次 = 174G 的读写；
# cp -al 只建目录项，**几乎不占空间也几乎不耗时**。orbax 是"写临时目录 → 原子改名"
# 落盘的，文件一旦就位就不再改写，所以硬链接是安全的：原件被 deleter 删掉只是减一个
# 链接数，_keep 里的 inode 照样在。（注意：这也意味着这些空间**不会**因为原件被删而释放，
# 这正是我们要的。）
#
# 用法: snapshot_ckpts.sh <PID> <CK目录> [轮询秒数]
set -uo pipefail

PID="${1:?需要训练进程 PID}"
CK="${2:?需要存档目录}"
POLL="${3:-60}"
# KEEP **必须放在存档根目录之外**。orbax 的 checkpoint_steps_paths() 会扫根目录里的每个
# 条目来判断是不是存档，源码注释明说 "this is not foolproof, and users should not add
# extra files to the checkpoint directory beyond what is done by CheckpointManager"。
# 实测（2026-09-19）_keep 放里面确实被正确忽略了（返回 [2000]），但既然作者明说不该这么干，
# 就没必要赌；而且放里面还会让"列出存档"的工具多出一个非数字条目。
KEEP="${KEEP_DIR:-$(dirname "$CK")/_keep_$(basename "$CK")}"
mkdir -p "$KEEP"

# 静置多久才算"写完了"。orbax 开着异步存档，且 create_directories_asynchronously=True，
# 所以目录名可能远早于内容就出现（实测 2000 那次：15:04:58 才 finalize 完）。若在写一半时
# 硬链接走，硬链接指向的还是**当时那个 inode**；orbax 之后若是"另写新文件再改名"，
# 我们留下的就是一份残缺档 —— 这种坏档混在里面，等评测时才发现就白等了。
STABLE=120

while kill -0 "$PID" 2>/dev/null; do
    now=$(date +%s)
    for d in "$CK"/*/; do
        b=$(basename "$d")
        case "$b" in _keep) continue ;; esac
        [ -e "$KEEP/$b" ] && continue
        # 闸一：落盘必备的元数据文件在位
        [ -e "$d/_CHECKPOINT_METADATA" ] || continue
        # 闸二：整棵树静置超过 STABLE 秒没被动过
        newest=$(find "$d" -newermt "@$((now - STABLE))" -print -quit 2>/dev/null)
        [ -z "$newest" ] || continue
        if cp -al "$d" "$KEEP/$b" 2>/dev/null; then
            echo "$(date '+%F %T') 留存 $b（硬链接，已静置 >${STABLE}s，$(du -sh "$d" 2>/dev/null | cut -f1)）"
        fi
    done
    sleep "$POLL"
done

echo "$(date '+%F %T') 训练进程 $PID 已退出，留存终止。_keep 内容:"
ls -1 "$KEEP" 2>/dev/null | sort -n | tr '\n' ' '; echo
du -sh --apparent-size "$KEEP" 2>/dev/null
