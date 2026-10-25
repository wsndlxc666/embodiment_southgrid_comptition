#!/usr/bin/env python
"""Stage 2「目标位置」oracle 数据集：18 原始维 + 3 维目标相对向量。

设计动机（与用户 2026-09-22 的新方案对齐）
==========================================
作废 d19/d22 那套 goal_dist / goal_y / cycle_dist / cycle_idx —— 它们是
「把 5 选 1 的离散身份换个编码再说一遍」，彼此高度共线。改成只问一件事：

    如果模型**确切知道当前目标在哪**，成功率会不会提升？

为什么这是 oracle（上界）
------------------------
target 取**每集自己实测的抓取点**（闭爪帧的 EE 位置），不是全局常量表。
工具在被抓起之前是静止的，它从第 0 帧起就在那儿 —— 所以这不是时间泄漏，
而是「一个完美的感知系统能给出的东西」。

于是结论是单边的：
  - oracle 无提升  -> 任何视觉/深度方案都不可能有用（它们只会更差），省掉整条 depth 路线
  - oracle 有提升  -> 才值得投入 perception

维度布局（state 21 维，action 保持 18 维不变）
--------------------------------------------
  0..17   原始 18 维（左右臂末端位姿 14 + 夹爪 4），逐字节沿用
  18..20  tgt_rel = target − ee   (base 系, 米)

用相对向量而不是绝对坐标，理由：ee 已经在 dims 7..9 里了，相对向量是直接可执行量，
且天然平移不变，模型不必额外学 base 系原点。

action **不动**（用户 Step 3 明确要求）。原始 action[f] == state[f+1] 逐帧成立，
直接整列沿用。

用法: build_stage2_oracle_dataset.py [集数, 默认 100]
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
DST = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_stage2_181ep"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100

EE_SLICE = slice(7, 10)      # 右臂末端 xyz (base 系)
GRIP_IDX = 16                # r_grip_inner_norm: 闭爪 1.0 / 开爪 0.048
GRIP_THRESH = 0.5
N_IN = 18
NEW_DIMS = ["tgt_rel_x", "tgt_rel_y", "tgt_rel_z"]
N_STATE = N_IN + len(NEW_DIMS)   # 21
N_ACT = N_IN                     # 18，不动


def find_cycles(state: np.ndarray) -> tuple[list[int], list[int]]:
    closed = state[:, GRIP_IDX] > GRIP_THRESH
    edges = np.where(np.diff(closed.astype(np.int8)) != 0)[0] + 1
    return ([int(f) for f in edges if closed[f]],
            [int(f) for f in edges if not closed[f]])


def build_target(grasp, release, n, picks, box):
    """逐帧的当前目标点（base 系）。轮次切点取在开爪帧之后一帧。"""
    target = np.empty((n, 3), dtype=np.float64)
    for k in range(5):
        a = 0 if k == 0 else release[k - 1] + 1
        r = release[k]
        target[a:r + 1] = picks[k]        # 第 k 轮盯着第 k 把（实测抓取点）
    target[release[-1] + 1:] = box        # 五把都放完 -> 目标切到箱子
    return target


def main() -> None:
    os.makedirs(os.path.join(DST, "data/chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(DST, "meta"), exist_ok=True)
    for cam in ("cam_head", "cam_wrist_r"):
        os.makedirs(os.path.join(DST, f"videos/chunk-000/observation.images.{cam}"), exist_ok=True)

    src_stats = [json.loads(l) for l in open(os.path.join(SRC, "meta/episodes_stats.jsonl"))]
    src_eps = [json.loads(l) for l in open(os.path.join(SRC, "meta/episodes.jsonl"))]

    new_eps, new_stats = [], []
    global_index = 0
    rel_lo, rel_hi = np.inf, -np.inf
    per_cycle_span = []

    print(f"源: {SRC}  取 {N_EP} 集")
    print(f"{'集':>4} {'帧':>6} {'轮次':>5}  {'tgt_rel 模长 范围(m)':<28}")
    print("-" * 60)

    for ep in range(N_EP):
        p = os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet")
        t = pq.read_table(p)
        state = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(t["action"].to_pylist(), dtype=np.float32)
        n = state.shape[0]
        if state.shape[1] != N_IN or action.shape[1] != N_IN:
            raise SystemExit(f"集{ep} 维数 {state.shape[1]}/{action.shape[1]}，预期 {N_IN}")

        # 自检：原始 action[t] == state[t+1]，**只到 n-2**。
        # 末帧没有 state[n]，采集端把它写成了 action[n-2] 的重复（实测差 3.3e-5），
        # 不是 state[n-1] —— 别把末帧一起比，会误报。
        if not np.allclose(action[:-1], state[1:], atol=1e-6):
            raise SystemExit(f"集{ep} 的 action != next state，前提被打破，停工")

        grasp, release = find_cycles(state)
        if len(grasp) != 5 or len(release) != 5:
            raise SystemExit(f"集{ep} 轮次异常: 闭{grasp} 开{release}")

        ee = state[:, EE_SLICE].astype(np.float64)
        picks = ee[grasp]                   # (5,3) 本集实测抓取点
        box = ee[release].mean(axis=0)      # (3,)  本集放盒位置
        target = build_target(grasp, release, n, picks, box)
        tgt_rel = (target - ee).astype(np.float32)

        m = np.linalg.norm(tgt_rel, axis=1)
        rel_lo, rel_hi = min(rel_lo, float(m.min())), max(rel_hi, float(m.max()))
        per_cycle_span.append(np.linalg.norm(np.diff(picks, axis=0), axis=1))

        state21 = np.concatenate([state, tgt_rel], axis=1).astype(np.float32)
        assert state21.shape[1] == N_STATE

        tbl = pa.table({
            "observation.state": pa.array([r.tolist() for r in state21], type=pa.list_(pa.float32())),
            "action": pa.array([r.tolist() for r in action], type=pa.list_(pa.float32())),
            "timestamp": t["timestamp"],
            "frame_index": t["frame_index"],
            "episode_index": pa.array([ep] * n, type=pa.int64()),
            "index": pa.array(list(range(global_index, global_index + n)), type=pa.int64()),
            "task_index": t["task_index"],
        })
        pq.write_table(tbl, os.path.join(DST, f"data/chunk-000/episode_{ep:06d}.parquet"))

        for cam in ("cam_head", "cam_wrist_r"):
            shutil.copy2(
                os.path.join(SRC, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"),
                os.path.join(DST, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"))

        st = {}
        for key, arr in (("observation.state", state21), ("action", action)):
            st[key] = {"min": arr.min(0).astype(float).tolist(),
                       "max": arr.max(0).astype(float).tolist(),
                       "mean": arr.mean(0).astype(float).tolist(),
                       "std": arr.std(0).astype(float).tolist(),
                       "count": [n]}
        for key in ("observation.images.cam_head", "observation.images.cam_wrist_r",
                    "timestamp", "frame_index", "episode_index", "task_index"):
            st[key] = src_stats[ep]["stats"][key]
        st["index"] = {"min": [global_index], "max": [global_index + n - 1],
                       "mean": [global_index + (n - 1) / 2.0],
                       "std": [float(np.std(np.arange(global_index, global_index + n)))],
                       "count": [n]}
        new_stats.append({"episode_index": ep, "stats": st})
        new_eps.append({"episode_index": ep, "tasks": src_eps[ep]["tasks"], "length": n})
        global_index += n
        print(f"{ep:>4} {n:>6} {len(grasp):>5}  [{m.min():.4f}, {m.max():.4f}]"
              f"   轮内跨集抖动(见下)", flush=True)

    # ================= meta =================
    info = json.load(open(os.path.join(SRC, "meta/info.json")))
    dim_names = info["features"]["observation.state"]["names"][0] + NEW_DIMS
    info["total_episodes"] = N_EP
    info["total_frames"] = int(global_index)
    info["total_videos"] = N_EP * 2
    info["splits"] = {"train": f"0:{N_EP}"}
    info["features"]["observation.state"]["shape"] = [N_STATE]
    info["features"]["observation.state"]["names"] = [dim_names]
    info["features"]["action"]["shape"] = [N_ACT]       # action 不动
    json.dump(info, open(os.path.join(DST, "meta/info.json"), "w"), indent=4)
    with open(os.path.join(DST, "meta/episodes.jsonl"), "w") as fh:
        for d in new_eps:
            fh.write(json.dumps(d) + "\n")
    with open(os.path.join(DST, "meta/episodes_stats.jsonl"), "w") as fh:
        for d in new_stats:
            fh.write(json.dumps(d) + "\n")
    shutil.copy2(os.path.join(SRC, "meta/tasks.jsonl"), os.path.join(DST, "meta/tasks.jsonl"))

    span = np.stack(per_cycle_span)
    meta = {
        "note": "Stage 2 oracle：state=21(原始18+目标相对向量3)，action=18 不变。"
                "target 取每集实测抓取点(工具静止，非时间泄漏)，故为感知上界。",
        "ee_slice": [7, 10],
        "grip_idx": GRIP_IDX,
        "grip_thresh": GRIP_THRESH,
        "new_dims": NEW_DIMS,
        "state_dim": N_STATE,
        "action_dim": N_ACT,
        "tgt_rel_range_m": [rel_lo, rel_hi],
        "target_rule": "按夹爪方波切 5 轮。第 k 轮 target=本集第 k 个闭爪帧的 EE 位置；"
                       "第 5 次开爪之后 target=本集 5 个开爪帧 EE 的均值(箱位)。"
                       "dim18..20 = target − ee (base 系)。",
        "index_advance": "推理端轮次靠**数开爪事件**推进，与训练同规则。",
    }
    json.dump(meta, open(os.path.join(DST, "tool_refs.json"), "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"state {N_STATE} 维 (18 + 3)   action {N_ACT} 维 (不变)")
    print(f"tgt_rel 模长范围: [{rel_lo:.4f}, {rel_hi:.4f}] m")
    print(f"集内相邻轮次实测间距: 中位 {np.median(span)*1000:.1f}mm"
          f"  p10 {np.percentile(span,10)*1000:.1f}  p90 {np.percentile(span,90)*1000:.1f}")
    print(f"落盘: {DST}   {N_EP} 集 / {global_index} 帧")


if __name__ == "__main__":
    main()
