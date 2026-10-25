#!/usr/bin/env python3
"""纯 GPU 压力测试 —— 完全不碰训练代码, 只做"大显存 + 满算力"的矩阵乘。

用途: 判断刚才训练崩溃到底是
  (a) 显卡/驱动本身扛不住满载 → 裸压力也会崩
  (b) 只是训练负载的特定模式触发 → 裸压力不崩

用法:
  uv run python gpu_stress_test.py [秒数]
默认 120 秒。每 10 秒打印一次 GPU 温度/功耗/利用率/显存。
"""
import os
import subprocess
import sys
import time

# 模拟训练时的显存占用比例(~22GB / 24GB)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp


def gpu_line() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip().replace("\n", " | ")
        return f"nvidia-smi 失败 rc={out.returncode}: {out.stderr.strip()}"
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi 异常: {e}"


def main() -> None:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print(f"JAX 设备: {jax.devices()}")
    print(f"开始纯 GPU 压力测试, 目标 {seconds} 秒 ...")

    # 一批大张量把显存顶到 ~18-20GB(bf16, 每元素 2 字节)
    bigs = [jnp.ones((8192, 8192, 16), dtype=jnp.bfloat16) for _ in range(8)]  # 8 * 2GiB = 16GiB
    x = jnp.ones((16384, 16384), dtype=jnp.bfloat16)  # 0.5GiB
    w = jnp.ones((16384, 16384), dtype=jnp.bfloat16)  # 0.5GiB

    t0 = time.time()
    last_report = 0
    try:
        while time.time() - t0 < seconds:
            y = x @ w
            y.block_until_ready()  # 强制等待 GPU 算完, 保证真的在满负载
            elapsed = int(time.time() - t0)
            if elapsed >= last_report + 10:
                print(f"  [{elapsed:4d}s] {gpu_line()}", flush=True)
                last_report = elapsed
        print(f"PASS: 纯 GPU 压力测试 {seconds}s 正常完成, 显卡没崩")
        print("结论: 崩溃不是裸 GPU 满载导致 → 问题在训练负载的特定模式(代码/驱动边界)")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: 纯 GPU 压力测试在第 {int(time.time()-t0)}s 崩溃: {e}", flush=True)
        print(f"崩溃瞬间 GPU 状态: {gpu_line()}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
