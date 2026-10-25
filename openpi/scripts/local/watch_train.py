#!/usr/bin/env python
"""盯一次训练：每跨过一个千步报一行，出任何错或进程死掉也报一行。

为什么不用 `tail -f | grep`：日志自己不会说话 —— 进程被 OOM killer 干掉、或者崩在
JIT 里，tail 那边是**一片安静**，而安静和"还在跑"长得一模一样。所以这里同时看三件事：
  1. 日志里有没有错误标记（OOM / Traceback / Xid / CRASH / 非有限 loss）
  2. diag jsonl 的进度（每 log_interval 一行，带逐维 loss）
  3. 进程还活着吗（kill -0）

输出行是 Monitor 的事件，所以只在**跨千步**、**出错**、**退出**这三种时候才打印。
给 Monitor 用：

  Monitor(command="python scripts/local/watch_train.py --pid-file /tmp/x.pid ...",
          description="d22idxI 20k 训练", persistent=True)

或前台手动跑（Ctrl-C 退出，不影响训练）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

# 失败标记。宁滥勿缺：漏掉一个就是一个"静默的成功"。
FAIL_PAT = re.compile(
    r"Traceback|RESOURCE_EXHAUSTED|CUDA_ERROR|out of memory|Out of memory|"
    r"Killed|OOM|\[CRASH\]|loss=nan|loss=inf|CANCELLED|Xid"
)
# 报错时顺带把这些上下文一起打出来（哪块内存、哪张卡）
CTX_PAT = re.compile(r"Allocator|ran out of memory|Xid|CUDA_ERROR|migrat")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_tail_line(path: str, start: int = 0) -> str:
    """读文件最后一行。训练在追加写，容忍读到一个半截行（下次再读）。

    start 是本轮训练开始时的文件字节数 —— diag jsonl 是**追加**写的，冒烟那轮的行
    还留在里面。不设起点的话，watcher 一上来就会把冒烟的最后一步当成"现在"报出来
    （而真正的 step 0 更小，之后再也不会触发 PROG，等于全程哑巴）。
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size <= start:
                return ""
            fh.seek(max(start, size - 65536))
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def scan_log(path: str) -> list[str]:
    try:
        out = subprocess.run(
            ["grep", "-aE", FAIL_PAT.pattern, path],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()][:4]


def scan_ctx(path: str) -> list[str]:
    try:
        out = subprocess.run(
            ["grep", "-aE", CTX_PAT.pattern, path],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()][-3:]


def fmt_step(rec: dict) -> str:
    step = int(rec["step"])
    loss = float(rec["loss"])
    gn = float(rec.get("grad_norm", float("nan")))
    parts = [f"step={step}", f"loss={loss:.4f}", f"|g|={gn:.3f}"]
    # 每 5000 步把索引/爪子那几维也带上，那是这一轮真正要看的
    if step % 5000 == 0:
        for d, nm in ((19, "goal_y"), (20, "cyc_dist"), (21, "cyc_idx"),
                      (16, "R爪"), (17, "R爪"),
                      (7, "R臂x")):
            if f"dim{d:02d}" in rec:
                parts.append(f"{nm}{d}={float(rec[f'dim{d:02d}']):.5f}")
    return " ".join(parts)


def main() -> int:
    # 被 Monitor 管道接走时 stdout 是块缓冲的 —— 不显式改行缓冲的话，事件会攒在
    # 缓冲区里一直不吐出来，表现就是"监视器一声不吭"，和"训练卡住了"长得一样。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--pid-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--every", type=int, default=1000, help="每多少步报一行")
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔（秒）")
    ap.add_argument("--grace", type=int, default=60, help="先等这么久再开始判定（躲开 JIT 编译）")
    a = ap.parse_args()

    try:
        pid = int(open(a.pid_file).read().strip())
    except (OSError, ValueError):
        print(f"START-ERR 读不到 PID: {a.pid_file}")
        return 1

    # 只认本轮训练写的行（diag 是追加的，冒烟那轮的还在里面）
    start = file_size(a.diag)
    log_start = file_size(a.log)

    print(f"WATCH pid={pid} diag起点={start}B log起点={log_start}B")
    time.sleep(a.grace)

    last_mark = -1
    while True:
        hits = scan_log(a.log)
        if hits:
            print("FAIL 训练日志出现错误标记：")
            for ln in hits:
                print("  " + ln[:220])
            for ln in scan_ctx(a.log):
                print("  ctx " + ln[:220])
            return 1

        line = read_tail_line(a.diag, start)
        rec = None
        if line:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rec = None  # 半截行，下轮再看
        if rec is not None:
            loss, gn = float(rec["loss"]), float(rec.get("grad_norm", 0.0))
            if not (math.isfinite(loss) and math.isfinite(gn)):
                print(f"FAIL loss/grad 非有限: {line[:220]}")
                return 1
            mark = int(rec["step"]) // a.every
            if mark > last_mark:
                last_mark = mark
                print(f"PROG {fmt_step(rec)}")

        if not alive(pid):
            tail = read_tail_line(a.diag, start)
            print(f"EXIT 训练进程 {pid} 已不在。diag 尾行: {tail[:220]}")
            print("     日志尾部：")
            try:
                out = subprocess.run(["tail", "-6", a.log], capture_output=True,
                                     text=True, timeout=30)
                for ln in out.stdout.splitlines():
                    print("  " + ln[:220])
            except (OSError, subprocess.SubprocessError):
                pass
            return 0

        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
