#!/usr/bin/env python
"""离线探针:拿训练集里"该闭爪"的真实帧喂给策略,对照模型预测 vs 真值。

用途:判定"不会闭爪"是模型没学会(训练侧)还是环境/映射问题(评测侧)。
输入:ckpt 路径(默认 43500)、集号。
输出:若干关键帧的 右爪/左爪 预测值 与 真值 对照。
"""
import os, subprocess, sys, tempfile
import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

BASE = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/dgl/lxc/openpi/checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/"
    "clear5_v1_l3090_lora/43500")
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# 每集自己的指令(训练时用的就是这条)
import json
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
         for l in open(f"{BASE}/meta/tasks.jsonl")}
t = pq.read_table(f"{BASE}/data/chunk-000/episode_{EP:06d}.parquet")
ti = int(np.array(t["task_index"].to_pylist())[0])
PROMPT = tasks[ti]
print(f"[probe] 集{EP} task_index={ti} prompt={PROMPT!r}", flush=True)

S = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
A = np.array(t["action"].to_pylist(), dtype=np.float32)


def frame(cam, idx):
    """用 ffmpeg 解出指定帧 (H,W,3) uint8"""
    out = os.path.join(tempfile.gettempdir(), f"probe_{cam}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", f"{BASE}/videos/chunk-000/observation.images.{cam}/episode_{EP:06d}.mp4",
         "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", out],
        check=True)
    from PIL import Image
    return np.array(Image.open(out).convert("RGB"), dtype=np.uint8)


from openpi.training import config as _config
from openpi.policies import policy_config

cfg = _config.get_config("pi05_g1_omnipicker_clear5_lora_s1500")
policy = policy_config.create_trained_policy(cfg, CKPT)
print("[probe] 策略已加载", flush=True)

# 关键帧:闭爪前 / 闭爪起点(114) / 闭合中 / 松开前 / 松开后
FRAMES = [50, 90, 110, 114, 130, 200, 300, 360, 380, 420]

print(f"\n{'帧':>5} {'GT右爪':>7} {'预测右爪':>8} {'GT左爪':>7} {'预测左爪':>8} "
      f"{'GT右臂Δ':>9} {'预测右臂Δ':>9}")
for f in FRAMES:
    head, wrist = frame("cam_head", f), frame("cam_wrist_r", f)
    payload = {"images": {"cam_head": head, "cam_wrist_r": wrist},
               "state": S[f], "prompt": PROMPT}
    a = np.asarray(policy.infer(payload)["actions"], dtype=np.float32)
    a0 = a[0, :18]
    # 右臂位移量(与当前 state 的差),用于判断手臂是否在动
    gd = np.abs(A[f, 7:10] - S[f, 7:10]).mean()
    pd = np.abs(a0[7:10] - S[f, 7:10]).mean()
    print(f"{f:>5} {A[f,16]:>7.3f} {a0[16]:>8.3f} {A[f,14]:>7.3f} {a0[14]:>8.3f} "
          f"{gd:>9.4f} {pd:>9.4f}", flush=True)
    print(f"      预测右爪两指={np.round(a0[16:18],3)} 预测状态爪={np.round(a0[14:18],3)}",
          flush=True)
print("\n[probe] 判读: 若\"预测右爪\"在 GT=1.0 的帧仍≈0.05 → 模型没学会闭爪(训练侧)")
print("[probe]       若\"预测右爪\"≈1.0 → 模型会闭,问题在评测/环境侧")
