#!/usr/bin/env python
"""把 clear5 数据集重建成 19 维版本:第19维 = 右臂末端到最近工具的欧氏距离。

- 源: data/g1_omnipicker_clear5_train (181集, 18维)
- 目标: data/g1_omnipicker_clear5_d19_100ep (前100集, 19维)
- 原始数据集只读不动。视频直接复制(帧内容不变)。

距离定义:
    tool_k = 该集第 k 次抓取时刻的右臂末端位置 (state[grasp_f, 7:10])
    d(f)   = min_k || state[f, 7:10] - tool_k ||
抓取瞬间 d≈0;机器人带着工具走开时 d 增大;接近下一件工具时 d 再次减小。
→ "靠近→抓取→离开→靠近" 的锯齿波形,正是要教给模型的相位信号。

action 沿用 action[f] == state[f+1] 的约定,故 action[:,18][f] = d(f+1)。
"""
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
DST = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d19_100ep"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
GRASP_FRAMES = [114, 514, 914, 1314, 1714]   # 每集 5 次闭爪的第一帧
EE_SLICE = slice(7, 10)                       # 右臂末端位置 (x,y,z), base 系


def main() -> None:
    # ---------- 目录骨架 ----------
    for sub in ("data/chunk-000",
                "videos/chunk-000/observation.images.cam_head",
                "videos/chunk-000/observation.images.cam_wrist_r",
                "meta"):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)

    src_stats = {}
    with open(os.path.join(SRC, "meta/episodes_stats.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            src_stats[d["episode_index"]] = d["stats"]

    src_eps = {}
    with open(os.path.join(SRC, "meta/episodes.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            src_eps[d["episode_index"]] = d

    new_eps, new_stats = [], []
    all_tool_refs = []
    global_index = 0
    dist_lo, dist_hi = 1e9, -1e9

    for ep in range(N_EP):
        src_pq = os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet")
        t = pq.read_table(src_pq)
        state = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
        action = np.array(t["action"].to_pylist(), dtype=np.float32)
        n = state.shape[0]

        # ---- 1. 该集的 5 个工具参考点 = 抓取时刻的末端位置 ----
        ee = state[:, EE_SLICE].astype(np.float64)          # (n,3)
        refs = np.stack([ee[f] for f in GRASP_FRAMES], axis=0)   # (5,3)
        all_tool_refs.append(refs)

        # ---- 2. 距离 = 到最近工具参考点的欧氏距离 ----
        # (n,5) 距离矩阵 → 取每帧最小值
        d = np.linalg.norm(ee[:, None, :] - refs[None, :, :], axis=2).min(axis=1)
        d = d.astype(np.float32)
        dist_lo, dist_hi = min(dist_lo, float(d.min())), max(dist_hi, float(d.max()))

        # ---- 3. 拼成 19 维 ----
        state19 = np.concatenate([state, d[:, None]], axis=1)
        d_next = np.concatenate([d[1:], d[-1:]], axis=0)     # action[f] = state[f+1]
        action19 = np.concatenate([action, d_next[:, None]], axis=1)

        # ---- 4. 写 parquet ----
        tbl = pa.table({
            "observation.state": pa.array([r.tolist() for r in state19], type=pa.list_(pa.float32())),
            "action": pa.array([r.tolist() for r in action19], type=pa.list_(pa.float32())),
            "timestamp": t["timestamp"],
            "frame_index": t["frame_index"],
            "episode_index": pa.array([ep] * n, type=pa.int64()),
            "index": pa.array(list(range(global_index, global_index + n)), type=pa.int64()),
            "task_index": t["task_index"],
        })
        pq.write_table(tbl, os.path.join(DST, f"data/chunk-000/episode_{ep:06d}.parquet"))

        # ---- 5. 视频直接复制(帧不变) ----
        for cam in ("cam_head", "cam_wrist_r"):
            s = os.path.join(SRC, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4")
            shutil.copy2(s, os.path.join(DST, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"))

        # ---- 6. 统计量 ----
        st = {}
        for key, arr in (("observation.state", state19), ("action", action19)):
            st[key] = {
                "min": arr.min(0).astype(float).tolist(),
                "max": arr.max(0).astype(float).tolist(),
                "mean": arr.mean(0).astype(float).tolist(),
                "std": arr.std(0).astype(float).tolist(),
                "count": [n],
            }
        # 视频与时间戳统计量原样沿用(视频逐字节相同)
        for key in ("observation.images.cam_head", "observation.images.cam_wrist_r",
                    "timestamp", "frame_index", "episode_index", "task_index"):
            st[key] = src_stats[ep][key]
        st["index"] = {
            "min": [global_index], "max": [global_index + n - 1],
            "mean": [global_index + (n - 1) / 2.0], "std": [float(np.std(np.arange(global_index, global_index + n)))],
            "count": [n],
        }
        new_stats.append({"episode_index": ep, "stats": st})
        new_eps.append({"episode_index": ep, "tasks": src_eps[ep]["tasks"], "length": n})
        global_index += n
        print(f"  集{ep:>3}: {n} 帧, 距离范围 [{float(d.min()):.4f}, {float(d.max()):.4f}] m", flush=True)

    # ---------- meta ----------
    info = json.load(open(os.path.join(SRC, "meta/info.json")))
    dim_names = info["features"]["observation.state"]["names"][0] + ["tool_dist"]
    info["total_episodes"] = N_EP
    info["total_frames"] = int(global_index)
    info["total_videos"] = N_EP * 2
    info["splits"] = {"train": f"0:{N_EP}"}
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [19]
        info["features"][key]["names"] = [dim_names]
    json.dump(info, open(os.path.join(DST, "meta/info.json"), "w"), indent=4)
    with open(os.path.join(DST, "meta/episodes.jsonl"), "w") as fh:
        for d in new_eps:
            fh.write(json.dumps(d) + "\n")
    with open(os.path.join(DST, "meta/episodes_stats.jsonl"), "w") as fh:
        for d in new_stats:
            fh.write(json.dumps(d) + "\n")
    shutil.copy2(os.path.join(SRC, "meta/tasks.jsonl"), os.path.join(DST, "meta/tasks.jsonl"))

    # ---------- 工具参考点(交给评测端用) ----------
    refs_mean = np.concatenate(all_tool_refs, 0).reshape(N_EP, 5, 3).mean(0)
    out = {
        "note": "右臂末端在 base 系下的抓取位(每件工具), 用于评测端计算第19维距离",
        "ee_slice": [7, 10],
        "tool_refs": refs_mean.tolist(),
        "dist_range_m": [dist_lo, dist_hi],
    }
    json.dump(out, open(os.path.join(DST, "tool_refs.json"), "w"), indent=2)

    print("\n=== 完成 ===")
    print(f"  目标: {DST}")
    print(f"  集数 {N_EP}, 总帧 {global_index}")
    print(f"  距离范围 [{dist_lo:.4f}, {dist_hi:.4f}] m")
    print("  5 个工具参考点(base 系):")
    for i, r in enumerate(refs_mean):
        print(f"    工具{i+1}: ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f})")


if __name__ == "__main__":
    main()
