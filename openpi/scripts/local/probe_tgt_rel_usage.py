#!/usr/bin/env python
"""反事实探针：Stage 2 的模型到底有没有**用**那 3 维 tgt_rel？

为什么需要它
============
训练 loss 降不降，不能区分这两件事：
  (a) 模型真的在读 tgt_rel，用它修正动作
  (b) 模型完全无视 tgt_rel，loss 降是因为它本来就能从图像里解出任务

这两者对外表现可能一模一样（因为 tgt_rel 与图像信息高度共线 —— 见
`probe_target_entropy.py`：目标位置的跨集 std 只有 8~11mm，而集内相邻工具间距
中位 103.8mm，信噪比 8~11 倍，所以"目标在哪"几乎就是"这是第几把工具"的重新编码）。
memory 里 `openpi-index-channel-is-weak` 记过同一件事：d22idx 的反事实干预
逼不出 idx，因为"读图"和"读 idx"给同一个 action，**相对梯度为零**。

判据因此必须是**干预**，不是看 loss：
  保持观测不变，只替换 tgt_rel，看预测动作变不变。
  变了 -> 模型确实在用这条通道
  不变 -> 模型无视它（那 Stage 2 就是白做，且 depth/perception 路线也可以砍）

三种干预
--------
  true   : 原始 tgt_rel（对照组）
  zero   : tgt_rel 归零（= "目标就在手边"）
  other  : 换成同一集里**另一把工具**的 tgt_rel（= "目标认错了"，位移量级 100mm+）

输出每个干预下的动作 L2 变化量，并按"是否在接近目标的阶段"分桶
（接近阶段 = 抓取前 40 帧内，此时目标信息最该起作用）。

用法: probe_tgt_rel_usage.py <ckpt_dir> [集号] [config名]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

BASE = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_stage2_181ep"
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "pi05_g1_omnipicker_clear5_stage2_lora"

if CKPT is None:
    ck_root = ("/home/dgl/lxc/openpi/checkpoints/"
               "pi05_g1_omnipicker_clear5_stage2_lora/clear5_stage2_oracle_181ep")
    steps = sorted(int(d) for d in os.listdir(ck_root) if d.isdigit())
    CKPT = os.path.join(ck_root, str(steps[-1]))
    print(f"[probe] 未指定 ckpt，自动取最新: {CKPT}")

N_IN, N_GRIP = 18, 16
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
         for l in open(f"{BASE}/meta/tasks.jsonl")}
t = pq.read_table(f"{BASE}/data/chunk-000/episode_{EP:06d}.parquet")
S = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
A = np.asarray(t["action"].to_pylist(), dtype=np.float32)
ti = int(np.asarray(t["task_index"].to_pylist())[0])
PROMPT = tasks[ti]
print(f"[probe] 集{EP}  prompt={PROMPT!r}  帧数={len(S)}  state {S.shape[1]} 维")


def frame(cam, idx):
    out = os.path.join(tempfile.gettempdir(), f"probe_tr_{cam}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                    f"{BASE}/videos/chunk-000/observation.images.{cam}/episode_{EP:06d}.mp4",
                    "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", out], check=True)
    from PIL import Image
    return np.array(Image.open(out).convert("RGB"), dtype=np.uint8)


# ---- 切轮次，拿到每把工具的 tgt_rel 基准 ----
closed = S[:, N_GRIP] > 0.5
grasp = (np.flatnonzero(~closed[:-1] & closed[1:]) + 1).tolist()
release = (np.flatnonzero(closed[:-1] & ~closed[1:]) + 1).tolist()
print(f"[probe] 闭爪帧={grasp}  开爪帧={release}")

# 第 k 轮（grasp[k-1]..release[k-1]）里 tgt_rel 的众数（应当只有 1~2 个取值：接近时在变，
# 但 target 绝对值恒定 → 用 target=rel+ee 的中位数当该轮的"目标点"）
EE = slice(7, 10)
cycle_target = []
for k, r in enumerate(release):
    a = 0 if k == 0 else release[k - 1] + 1
    seg = S[a:r + 1, 18:21] + S[a:r + 1, EE]
    cycle_target.append(np.median(seg, axis=0))
cycle_target = np.stack(cycle_target)
print("[probe] 每轮反推出的目标点 (base 系):")
for k, c in enumerate(cycle_target):
    print(f"   轮{k+1}: [{c[0]:+.4f} {c[1]:+.4f} {c[2]:+.4f}]")

# 每轮中段的帧（避开起步/停稳，取轮次中段 1/4~3/4）作为探针帧
probe_frames, probe_cycle = [], []
for k, r in enumerate(release):
    a = 0 if k == 0 else release[k - 1] + 1
    lo, hi = a + (r - a) // 4, a + 3 * (r - a) // 4
    idx = list(range(lo, hi, max(1, (hi - lo) // 8)))
    probe_frames += idx
    probe_cycle += [k] * len(idx)
probe_frames = np.array(probe_frames)
probe_cycle = np.array(probe_cycle)
print(f"[probe] 探针帧 {len(probe_frames)} 个（每轮 8 帧）")

# 图像只解一次，三种干预共用（这样差异只可能来自 tgt_rel）
imgs = {f: (frame("cam_head", f), frame("cam_wrist_r", f)) for f in probe_frames}
print("[probe] 图像已解码", flush=True)

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

policy = policy_config.create_trained_policy(_config.get_config(CONFIG), CKPT)
print("[probe] 策略已加载", flush=True)


def infer(f, rel):
    st = S[f].copy()
    st[18:21] = rel
    head, wrist = imgs[f]
    payload = {"images": {"cam_head": head, "cam_wrist_r": wrist},
               "state": st, "prompt": PROMPT}
    return np.asarray(policy.infer(payload)["actions"], dtype=np.float32)[0, :N_IN]


rows = []
for f, k in zip(probe_frames, probe_cycle, strict=True):
    true_rel = S[f, 18:21]
    ee = S[f, EE]
    acts = {
        "true": infer(f, true_rel),
        "zero": infer(f, np.zeros(3, dtype=np.float32)),
        # 换成另一把工具（取轮次相隔最远的那个，位移最大）
        "other": infer(f, (cycle_target[(k + 3) % 5] - ee).astype(np.float32)),
    }
    rows.append((f, k, acts))

print(f"\n{'帧':>5} {'轮':>3} | {'|zero−true|':>12} {'|other−true|':>13} "
      f"| {'右臂Δ(zero)':>12} {'右臂Δ(other)':>13}")
print("-" * 78)
d_zero, d_other, d_arm_zero, d_arm_other = [], [], [], []
for f, k, acts in rows:
    a = acts["true"]
    z = np.linalg.norm(acts["zero"] - a)
    o = np.linalg.norm(acts["other"] - a)
    az = np.linalg.norm((acts["zero"] - a)[7:10])
    ao = np.linalg.norm((acts["other"] - a)[7:10])
    d_zero.append(z); d_other.append(o); d_arm_zero.append(az); d_arm_other.append(ao)
    print(f"{f:>5} {k+1:>3} | {z:>12.5f} {o:>13.5f} | {az:>12.5f} {ao:>13.5f}")

d_zero, d_other = np.array(d_zero), np.array(d_other)
d_arm_other = np.array(d_arm_other)
print("\n=== 汇总 ===")
print(f"  全动作 L2  干预 zero : 中位 {np.median(d_zero):.5f}  最大 {d_zero.max():.5f}")
print(f"  全动作 L2  干预 other: 中位 {np.median(d_other):.5f}  最大 {d_other.max():.5f}")
print(f"  右臂 L2    干预 other: 中位 {np.median(d_arm_other):.5f}  最大 {d_arm_other.max():.5f}")
# 参照尺度：相邻帧真值动作的自然变化量
nat = np.linalg.norm(np.diff(A[probe_frames, :N_IN], axis=0), axis=1)
print(f"  参照: 真值动作逐帧自然变化 中位 {np.median(nat):.5f}")

print("\n=== 判读（判据事先写死）===")
med = np.median(d_other)
ref = np.median(nat)
if med < 0.05 * ref:
    print(f"  other 干预扰动量 {med:.5f} < 自然变化的 5% ({0.05*ref:.5f})")
    print("  -> **模型基本无视 tgt_rel**。Stage 2 没起到作用，")
    print("     且 oracle 都无效 => 任何 depth/vision 估位方案只会更差，整条路线可砍。")
elif med < 0.5 * ref:
    print(f"  other 干预扰动量 {med:.5f} 在自然变化的 5%~50% 之间 ({ref:.5f})")
    print("  -> **通道通了但很弱**。模型读到了 tgt_rel，但它对动作的影响")
    print("     小于环境本身带来的变化 —— 提升会很小且难在成功率上测出来。")
else:
    print(f"  other 干预扰动量 {med:.5f} >= 自然变化的 50% ({ref:.5f})")
    print("  -> **模型确实在用 tgt_rel**。这才值得继续投入 perception")
    print("     （此时再谈 depth / 视觉估位怎么把这 3 维算出来）。")
