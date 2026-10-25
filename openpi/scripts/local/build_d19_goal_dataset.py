#!/usr/bin/env python
"""把 clear5 训练集重建成 19 维版本 v2：第19维 = 右臂末端到最近"关键点"的距离。

与 v1 (`build_d19_dataset.py`) 的区别
------------------------------------
v1: d(f) = min_k ‖ee(f) − tool_k‖          —— 只对 5 个抓取点取最近
     问题1 取 min 抹掉了"是哪一把"，argmin 不跟着从左到右走
     问题2 搬运/放盒时这个数只反映"盒子碰巧离哪把工具最近"，没有意义
     问题3 实测取值几乎二值：73.1% 的帧集中在 0.25~0.35，只有 18.8% 的帧 d<0.05

v2: d(f) = min( min_k ‖ee(f) − tool_k‖ , ‖ee(f) − box‖ )
     把"盒子落点"也加进关键点集合。于是
       · 闭爪时 → 0（手在工具上）
       · 放进盒子时 → 0（手在盒子上）  ← v1 在这里是 0.27，不收敛
       · 每集归零 10 次（5 抓 + 5 放），相位信息翻倍
       · 全程连续，无跳变（对比"周期版"定义在换目标时会有 +0.05~+0.13 的阶跃）

关键点常量：两遍扫描
------------------
第一遍读全部 episode，求出全局的 5 个抓取点 + 1 个盒子落点（跨集取均值）；
第二遍用这组**固定常量**写数据。
这样训练时数据里的 d 和推理时客户端算的 d 是**同一个函数**，不留 v1 那种
"训练用本集自己的参考点、推理用均值"的 ~1cm 错配。

原始数据集只读不动。视频直接复制（帧内容不变）。

用法:  build_d19_goal_dataset.py [集数, 默认 100]
"""
import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
DST = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d19_goal_100ep"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100

EE_SLICE = slice(7, 10)      # 右臂末端 xyz (base 系)
GRIP_IDX = 16                # r_grip_inner_norm: 闭爪 1.0 / 开爪 0.048
GRIP_THRESH = 0.5
DIM_NAME = "goal_dist"


def read_state(ep: int) -> np.ndarray:
    p = os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet")
    return np.array(pq.read_table(p)["observation.state"].to_pylist(), dtype=np.float32)


def find_cycles(state: np.ndarray) -> tuple[list[int], list[int]]:
    """从夹爪方波直接求出 (闭爪帧, 开爪帧)，不做硬编码。"""
    closed = state[:, GRIP_IDX] > GRIP_THRESH
    edges = np.where(np.diff(closed.astype(np.int8)) != 0)[0] + 1
    return ([int(f) for f in edges if closed[f]],
            [int(f) for f in edges if not closed[f]])


def main() -> None:
    # ================= 第一遍：求全局关键点常量 =================
    print("=" * 72)
    print("第一遍：扫描全部 episode，求全局关键点常量")
    print("=" * 72)

    all_tools, all_boxes, n_frames = [], [], []
    for ep in range(N_EP):
        s = read_state(ep)
        grasp, release = find_cycles(s)
        if len(grasp) != 5 or len(release) != 5:
            raise SystemExit(f"集{ep} 的周期数异常: 闭爪{len(grasp)} 开爪{len(release)}")
        ee = s[:, EE_SLICE]
        all_tools.append(ee[grasp])                 # (5,3)
        all_boxes.append(ee[release].mean(axis=0))  # (3,)
        n_frames.append(len(s))

    tools = np.stack(all_tools)          # (N,5,3)
    boxes = np.stack(all_boxes)          # (N,3)
    TOOL_REF = tools.mean(axis=0)        # (5,3) 全局常量
    BOX_REF = boxes.mean(axis=0)         # (3,)  全局常量
    GOALS = np.concatenate([TOOL_REF, BOX_REF[None, :]], axis=0)   # (6,3)

    print(f"  集数 {N_EP}, 每集帧数 {min(n_frames)}~{max(n_frames)}")
    print(f"\n  跨集离散度 (衡量数据有多'一致'):")
    print(f"    抓取点: 平均偏离 {np.linalg.norm(tools - TOOL_REF[None], axis=2).mean() * 100:.2f} cm")
    print(f"    盒子点: 平均偏离 {np.linalg.norm(boxes - BOX_REF[None], axis=1).mean() * 100:.2f} cm")
    print(f"\n  全局关键点 (base 系, 米):")
    for i in range(5):
        print(f"    工具{i + 1}: ({TOOL_REF[i, 0]:+.4f}, {TOOL_REF[i, 1]:+.4f}, {TOOL_REF[i, 2]:+.4f})")
    print(f"    盒子  : ({BOX_REF[0]:+.4f}, {BOX_REF[1]:+.4f}, {BOX_REF[2]:+.4f})")

    # ================= 第二遍：写数据 =================
    print()
    print("=" * 72)
    print("第二遍：用固定常量写 19 维数据")
    print("=" * 72)

    for sub in ("data/chunk-000",
                "videos/chunk-000/observation.images.cam_head",
                "videos/chunk-000/observation.images.cam_wrist_r",
                "meta"):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)

    src_stats = {json.loads(l)["episode_index"]: json.loads(l)["stats"]
                 for l in open(os.path.join(SRC, "meta/episodes_stats.jsonl"))}
    src_eps = {json.loads(l)["episode_index"]: json.loads(l)
               for l in open(os.path.join(SRC, "meta/episodes.jsonl"))}

    new_eps, new_stats = [], []
    global_index = 0
    d_lo, d_hi, max_jump = 1e9, -1e9, 0.0

    for ep in range(N_EP):
        t = pq.read_table(os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet"))
        state = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
        action = np.array(t["action"].to_pylist(), dtype=np.float32)
        n = state.shape[0]
        ee = state[:, EE_SLICE].astype(np.float64)

        d = np.linalg.norm(ee[:, None, :] - GOALS[None, :, :], axis=2).min(axis=1)
        d = d.astype(np.float32)
        d_lo, d_hi = min(d_lo, float(d.min())), max(d_hi, float(d.max()))
        max_jump = max(max_jump, float(np.abs(np.diff(d)).max()))

        state19 = np.concatenate([state, d[:, None]], axis=1)
        d_next = np.concatenate([d[1:], d[-1:]], axis=0)      # action[f] == state[f+1]
        action19 = np.concatenate([action, d_next[:, None]], axis=1)

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

        for cam in ("cam_head", "cam_wrist_r"):
            shutil.copy2(
                os.path.join(SRC, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"),
                os.path.join(DST, f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"))

        st = {}
        for key, arr in (("observation.state", state19), ("action", action19)):
            st[key] = {"min": arr.min(0).astype(float).tolist(),
                       "max": arr.max(0).astype(float).tolist(),
                       "mean": arr.mean(0).astype(float).tolist(),
                       "std": arr.std(0).astype(float).tolist(),
                       "count": [n]}
        for key in ("observation.images.cam_head", "observation.images.cam_wrist_r",
                    "timestamp", "frame_index", "episode_index", "task_index"):
            st[key] = src_stats[ep][key]
        st["index"] = {"min": [global_index], "max": [global_index + n - 1],
                       "mean": [global_index + (n - 1) / 2.0],
                       "std": [float(np.std(np.arange(global_index, global_index + n)))],
                       "count": [n]}
        new_stats.append({"episode_index": ep, "stats": st})
        new_eps.append({"episode_index": ep, "tasks": src_eps[ep]["tasks"], "length": n})
        global_index += n
        print(f"  集{ep:>3}: {n} 帧, d ∈ [{float(d.min()):.4f}, {float(d.max()):.4f}]", flush=True)

    # ================= meta =================
    info = json.load(open(os.path.join(SRC, "meta/info.json")))
    dim_names = info["features"]["observation.state"]["names"][0] + [DIM_NAME]
    info["total_episodes"] = N_EP
    info["total_frames"] = int(global_index)
    info["total_videos"] = N_EP * 2
    info["splits"] = {"train": f"0:{N_EP}"}
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [19]
        info["features"][key]["names"] = [dim_names]
    json.dump(info, open(os.path.join(DST, "meta/info.json"), "w"), indent=4)
    with open(os.path.join(DST, "meta/episodes.jsonl"), "w") as fh:
        for d_ in new_eps:
            fh.write(json.dumps(d_) + "\n")
    with open(os.path.join(DST, "meta/episodes_stats.jsonl"), "w") as fh:
        for d_ in new_stats:
            fh.write(json.dumps(d_) + "\n")
    shutil.copy2(os.path.join(SRC, "meta/tasks.jsonl"), os.path.join(DST, "meta/tasks.jsonl"))

    # ================= 关键点常量(交给评测端) =================
    out = {
        "note": "右臂末端在 base 系下的关键点(5 个工具抓取点 + 1 个盒子落点)。"
                "评测端算第19维: d = min_j ‖ee − goals[j]‖，j 遍历 6 个点。",
        "ee_slice": [7, 10],
        "tool_refs": TOOL_REF.tolist(),
        "box_ref": BOX_REF.tolist(),
        "goals": GOALS.tolist(),
        "dist_range_m": [d_lo, d_hi],
        "build": "v2-goalset (两遍扫描, 全局常量)",
    }
    json.dump(out, open(os.path.join(DST, "tool_refs.json"), "w"), indent=2)

    print()
    print("=" * 72)
    print("完成")
    print("=" * 72)
    print(f"  目标: {DST}")
    print(f"  集数 {N_EP}, 总帧 {global_index}")
    print(f"  d 范围 [{d_lo:.4f}, {d_hi:.4f}] m")
    print(f"  相邻帧最大跳变 {max_jump:.4f} m   (100集同口径: v1 min_k 0.0850, 周期版 0.1304)")
    print(f"  关键点常量已写入 {DST}/tool_refs.json")


if __name__ == "__main__":
    main()
