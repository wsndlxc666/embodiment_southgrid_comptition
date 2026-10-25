#!/usr/bin/env python
"""反事实探针·第二轮：tgt_rel 的耦合**有没有方向**？

第一轮（probe_tgt_rel_usage.py）只测了扰动的**大小**，结论落在"很弱（12~21%）"这一档，
而且随步数单调下降。但那条判据有个漏洞：

    它拿"数据自身的自然逐帧变化"（0.233）当尺子。可是**任何**输入 token 变了，
    输出本来就会晃 —— 12% 完全可能只是通用抖动，跟 tgt_rel 无关。

大小测不出"在用"和"抖动"的区别，**方向**可以：

    如果模型真在用 tgt_rel 定目标，把目标挪 100mm，它预测的右臂落点
    必须**朝那个方向**偏。cos(Δ动作, Δ目标) 应显著为正。
    如果只是抖动，Δ动作的方向与 Δ目标无关，cos ≈ 0、正负各半。

三个干预
--------
  true    原始 state（对照）
  other   tgt_rel 换成同集里另一把工具的（位移 61~136mm，与第一轮完全一致，可对齐）
  zero    tgt_rel 归零
  ee_ctrl **阳性对照**：把 state[7:10]（EE 位置）平移 +50mm(x)。
          模型是跟踪策略，action[t]==state[t+1]，所以它的预测落点**必须**跟着走
          （cos≈1, gain≈1）。这条用来证明"本探针测得出方向耦合" ——
          否则 tgt_rel 测出 cos≈0 也可能是探针本身坏掉。

输出
----
  1) 逐维 Δ动作（other 干预）—— 看响应落在哪些维上。目标信息最该影响 dims 7:10。
  2) 沿 action chunk 的 t=0,4,9,14,19，cos 与 gain：
       cos  = <Δaction[t][7:10], Δtarget> / (|·||·|)
       gain = |Δaction[t][7:10]| / |Δtarget|
     chunk 越靠后，若模型在朝目标收敛，gain 应越大、cos 应越稳。

判据（**写死在跑之前**）
------------------------
先看阳性对照：
  ee_ctrl 的 median cos < 0.7  ->  探针本身测不出方向，本次结论一律作废，回去查原因。

再看 tgt_rel（other 干预），取所有 t 中 cos 最大的那个：
  median cos >= 0.30 且 同向帧占比 >= 70%      ->  H1「模型在用 tgt_rel 定目标」
  |median cos| < 0.15 或 同向帧占比 <= 60%     ->  H0「方向无关，第一轮的 12% 是通用抖动」
  其余                                          ->  弱方向性（沿用第一轮的"很弱"结论）

用法: probe_tgt_rel_direction.py <ckpt_dir> [集号] [config名]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

BASE = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_stage2_181ep"
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "pi05_g1_omnipicker_clear5_stage2_lora"

N_IN, N_GRIP = 18, 16
EE = slice(7, 10)
TGT = slice(18, 21)
EE_SHIFT = np.array([0.0, 0.05, 0.0], dtype=np.float32)   # 阳性对照：EE 位置 +50mm(y)
STEPS = (0, 4, 9, 14, 19)

tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
         for l in open(f"{BASE}/meta/tasks.jsonl")}
t = pq.read_table(f"{BASE}/data/chunk-000/episode_{EP:06d}.parquet")
S = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
A = np.asarray(t["action"].to_pylist(), dtype=np.float32)
PROMPT = tasks[int(np.asarray(t["task_index"].to_pylist())[0])]
names = json.load(open(f"{BASE}/meta/info.json"))["features"]["observation.state"]["names"][0]
print(f"[probe2] ckpt={CKPT}")
print(f"[probe2] 集{EP} prompt={PROMPT!r} 帧数={len(S)} state {S.shape[1]} 维")
print(f"[probe2] dims 18:21 = {names[18:21]}   dims 7:10 = {names[7:10]}")


def frame(cam, idx):
    out = os.path.join(tempfile.gettempdir(), f"probe2_{cam}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                    f"{BASE}/videos/chunk-000/observation.images.{cam}/episode_{EP:06d}.mp4",
                    "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", out], check=True)
    from PIL import Image
    return np.array(Image.open(out).convert("RGB"), dtype=np.uint8)


closed = S[:, N_GRIP] > 0.5
release = (np.flatnonzero(closed[:-1] & ~closed[1:]) + 1).tolist()

cycle_target = []
for k, r in enumerate(release):
    a = 0 if k == 0 else release[k - 1] + 1
    cycle_target.append(np.median(S[a:r + 1, TGT] + S[a:r + 1, EE], axis=0))
cycle_target = np.stack(cycle_target)

probe_frames, probe_cycle = [], []
for k, r in enumerate(release):
    a = 0 if k == 0 else release[k - 1] + 1
    lo, hi = a + (r - a) // 4, a + 3 * (r - a) // 4
    idx = list(range(lo, hi, max(1, (hi - lo) // 8)))
    probe_frames += idx
    probe_cycle += [k] * len(idx)
probe_frames = np.array(probe_frames)
probe_cycle = np.array(probe_cycle)
print(f"[probe2] 探针帧 {len(probe_frames)} 个")

imgs = {f: (frame("cam_head", f), frame("cam_wrist_r", f)) for f in probe_frames}
print("[probe2] 图像已解码", flush=True)

def cos(d, u):
    """两个向量的余弦。任一为零向量时返回 0（当作"没方向"）。"""
    nd, nu = float(np.linalg.norm(d)), float(np.linalg.norm(u))
    return float(d @ u / (nd * nu)) if nd > 1e-9 and nu > 1e-9 else 0.0


from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

policy = policy_config.create_trained_policy(_config.get_config(CONFIG), CKPT)
print("[probe2] 策略已加载", flush=True)


def infer(f, st):
    """返回整条 action chunk (20, 18)。

    ⚠ 第一轮 probe_tgt_rel_usage.py 写的是 `[0, :N_IN]`（只取 chunk 第一步），
    那只够回答"扰动多大"。本探针要沿 chunk 看方向，所以必须保留全部 20 步 ——
    照抄那个下标会得到 (18,)，下一步 `a[tt, EE]` 直接 IndexError。
    """
    head, wrist = imgs[f]
    return np.asarray(policy.infer(
        {"images": {"cam_head": head, "cam_wrist_r": wrist},
         "state": st, "prompt": PROMPT})["actions"], dtype=np.float32)[..., :N_IN]


# ---------------- 逐帧跑四种干预 ----------------
per_dim = []          # other 干预的逐维 |Δ|
cos_by_step = {t: [] for t in STEPS}          # other
cos_zero = {t: [] for t in STEPS}             # zero
cos_ctrl = {t: [] for t in STEPS}             # 阳性对照
gain_by_step = {t: [] for t in STEPS}         # other
gain_ctrl = {t: [] for t in STEPS}            # 阳性对照
dtgt_norm = []

for f, k in zip(probe_frames, probe_cycle, strict=True):
    st = S[f].copy()
    a_true = infer(f, st)

    rel_other = (cycle_target[(k + 3) % 5] - st[EE]).astype(np.float32)
    dtgt = rel_other - st[TGT]                       # 目标位移（同一帧，ee 抵消）
    dtgt_norm.append(np.linalg.norm(dtgt))

    st_o = st.copy(); st_o[TGT] = rel_other
    a_other = infer(f, st_o)

    st_z = st.copy(); st_z[TGT] = 0.0
    a_zero = infer(f, st_z)

    st_c = st.copy(); st_c[EE] = st[EE] + EE_SHIFT
    a_ctrl = infer(f, st_c)

    per_dim.append(np.abs(a_other - a_true).mean(axis=0))

    for tt in STEPS:
        d_o = a_other[tt, EE] - a_true[tt, EE]
        d_z = a_zero[tt, EE] - a_true[tt, EE]
        d_c = a_ctrl[tt, EE] - a_true[tt, EE]

        cos_by_step[tt].append(cos(d_o, dtgt))
        cos_zero[tt].append(cos(d_z, dtgt))
        cos_ctrl[tt].append(cos(d_c, EE_SHIFT))
        gain_by_step[tt].append(np.linalg.norm(d_o) / (np.linalg.norm(dtgt) + 1e-9))
        gain_ctrl[tt].append(np.linalg.norm(d_c) / (np.linalg.norm(EE_SHIFT) + 1e-9))

print(f"\n目标位移 |Δtarget| 中位 {np.median(dtgt_norm)*1000:.1f} mm")

print("\n=== 1) 逐维 Δ动作（other 干预，全 chunk 平均）===")
pd_ = np.mean(per_dim, axis=0)
for i, v in enumerate(pd_):
    mark = "  <- tgt_rel 该影响这里" if i in (7, 8, 9) else ""
    print(f"  dim{i:>2} {names[i]:<22} {v:.5f}{mark}")
print(f"  dims 7:10 合计占比 {pd_[7:10].sum()/pd_.sum()*100:.1f}%"
      f"   （若均匀分布应约 {3/18*100:.1f}%）")


def summarize(tag, cosd, gaind=None):
    print(f"\n=== {tag} ===")
    print(f"{'chunk t':>8} {'median cos':>11} {'同向帧占比':>10} "
          f"{'median gain':>12} {'P25 cos':>9} {'P75 cos':>9}")
    for tt in STEPS:
        c = np.array(cosd[tt])
        row = (f"{tt:>8} {np.median(c):>11.3f} {np.mean(c > 0)*100:>9.0f}% ")
        if gaind is not None:
            row += f"{np.median(gaind[tt]):>12.3f} "
        else:
            row += f"{'—':>12} "
        row += f"{np.percentile(c,25):>9.3f} {np.percentile(c,75):>9.3f}"
        print(row)
    return {tt: np.median(np.array(cosd[tt])) for tt in STEPS}


med_ctrl = summarize("阳性对照 ee_ctrl（EE 位置 +50mm，模型**应该**跟）", cos_ctrl, gain_ctrl)
med_o = summarize("测试 other（tgt_rel 换成另一把工具）", cos_by_step, gain_by_step)
med_z = summarize("测试 zero（tgt_rel 归零）", cos_zero)

print("\n=== 判读（判据写死在脚本 docstring 里，跑之前就定好）===")
best_ctrl = max(med_ctrl.values())
print(f"[闸] 阳性对照最好 median cos = {best_ctrl:.3f}（阈值 0.70）")
if best_ctrl < 0.70:
    print("  ✗ 阳性对照没过 —— 本探针测不出方向耦合，**本次结论一律作废**，先查探针。")
    sys.exit(2)
print("  ✓ 阳性对照通过：探针能测出方向耦合，下面的 cos 可信。")

best_t = max(med_o, key=lambda k: med_o[k])
bc = med_o[best_t]
frac = np.mean(np.array(cos_by_step[best_t]) > 0)
print(f"\n[判] other 干预最好一档 t={best_t}: median cos={bc:.3f} 同向帧占比={frac*100:.0f}%")
bz = max(med_z, key=lambda k: med_z[k])
print(f"     zero 干预最好一档 t={bz}: median cos={med_z[bz]:.3f} "
      f"同向帧占比={np.mean(np.array(cos_zero[bz])>0)*100:.0f}%")
if bc >= 0.30 and frac >= 0.70:
    print("  -> **H1 成立：模型确实在朝目标方向修正**。tgt_rel 通道有效，值得投入 perception。")
elif abs(bc) < 0.15 or frac <= 0.60:
    print("  -> **H0 成立：Δ动作方向与 Δ目标无关**。第一轮那 12% 是通用抖动，")
    print("     不是『在用目标』。oracle 都逼不出方向 => depth/视觉估位只会更差，整条可砍。")
else:
    print("  -> **弱方向性**：沿用第一轮『通道通了但很弱』的结论，")
    print("     提升会很小且难在成功率上测出来。")
