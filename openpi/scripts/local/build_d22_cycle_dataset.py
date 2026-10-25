#!/usr/bin/env python
"""clear5_train -> 22 维 v3「周期感知」版：在 d19_goal(v2) 基础上再加 3 维。

v2 已有的（19 维，保持不变）
--------------------------
  0..17  原始 18 维（左右臂位姿 14 + 夹爪 4）
  18     goal_dist = min_j ‖ee − goals[j]‖, j 遍历 {5 个抓取点, 盒子}

  第 18 维**原样保留**：它跟 9000 步那个 ckpt 训的是同一个量、同一套归一化，
  留着能让热启动后前 19 个数字对模型保持同样的含义。换掉它没有好处。

v3 新增（19..21）
----------------
  19  goal_y      : **当前目标点**的 y（base 系，米）
  20  cycle_dist  : ‖ee − 当前目标点‖
  21  cycle_idx   : 1..5 = 正在处理第几把；6 = 五把都放完了（可以停）

「当前目标点」的定义（周期感知，修掉 v2 的 argmin 歧义）
------------------------------------------------------
按夹爪方波把每集切成 5 轮，第 k 轮（k=1..5，0 基）：

    [上一轮开爪, 第k次开爪]  ->  目标 = TOOL_REF[k]   （整个第 k 轮都盯着这把）
    [第 5 次开爪 + 1, 集末]  ->  目标**冻结**在第 5 把, idx = 6   （五把都放完，可以停）

于是 cycle_dist 在**每轮的闭爪帧上精确归零** —— 那个 0 就是「现在该合爪」的信号。
对比 v2 的 min_j：那个数在搬运途中会「就近认领」到旁边另一把工具上，
读数只反映「盒子碰巧离谁近」，这正是 v1/v2 都修不掉的那个歧义。

为什么 idx=6 不切到盒子（试过，弃用）
----------------------------------
idx=6 时若把目标换成 BOX_REF，读数会从 ‖盒子−第5把‖ = **0.655 m** 直接掉到 0 ——
第 5 把是所有工具里离盒子最远的（y 差 0.65 m）。实测最大跳变被这一处拉到 0.6405 m，
是 v2 的 0.0728 m 的 9 倍。而 action[f] == state[f+1]，模型得逐帧预测这个跳变。
现在冻在第 5 把上，最大跳变只剩轮与轮之间的 ~0.11 m，和 v2 同量级。
「我已经在盒子上了」这个信息模型从自己的手臂 state(7..10) 和画面里本来就有，
不需要位置维再讲一遍。

为什么只加 y，不加 xyz
----------------------
实测 5 把工具在 base 系下的位置（见 tool_refs.json）：

    y 跨度 407.9 mm，相邻工具差 77~131 mm    <- 5~10 倍于布局抖动，唯一可靠判据
    x 跨度  44.7 mm，相邻工具差 11.5~33.1 mm <- 落在抖动带内(±8mm, 12%概率±40mm)，读不出身份
    z 跨度  19.8 mm，相邻工具差  3.5~17.1 mm <- 就是工具厚度

所以 x 是**噪声维**（差异被抖动淹没），z 是**小跨度维**（跨度 19.8mm，
和左臂那几维 1.4~18mm 同一区间 —— 那几维曾吃掉 63.8% 的 loss，是已经踩过的坑）。
只有 y 是干净的判据。

关于 action
----------
action 同步扩到 22 维（沿用 v2 的做法，action[f] == state[f+1]）。
这 3 维在推理时会被 G1OmnipickerOutputs 按 action_dim 截掉，**不参与控制**，
作用只是辅助训练信号（逼模型编码"现在是第几轮"）。

原始数据集只读不动。视频直接复制（帧内容不变）。

用法:  build_d22_cycle_dataset.py [集数, 默认 100]
"""

import json
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_train"
DST = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100

EE_SLICE = slice(7, 10)      # 右臂末端 xyz (base 系)
GRIP_IDX = 16                # r_grip_inner_norm: 闭爪 1.0 / 开爪 0.048
GRIP_THRESH = 0.5
DIM18 = "goal_dist"
NEW_DIMS = ["goal_y", "cycle_dist", "cycle_idx"]
N_IN = 18                    # 原始 state/action 维数
N_OUT = N_IN + 1 + len(NEW_DIMS)   # = 22


def read_state(ep: int) -> np.ndarray:
    p = os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet")
    return np.array(pq.read_table(p)["observation.state"].to_pylist(), dtype=np.float32)


def find_cycles(state: np.ndarray) -> tuple[list[int], list[int]]:
    """从夹爪方波直接求出 (闭爪帧, 开爪帧)，不做硬编码。"""
    closed = state[:, GRIP_IDX] > GRIP_THRESH
    edges = np.where(np.diff(closed.astype(np.int8)) != 0)[0] + 1
    return ([int(f) for f in edges if closed[f]],
            [int(f) for f in edges if not closed[f]])


def build_cycle_targets(grasp, release, n, tool_ref):
    """逐帧的「当前目标点」和周期号。规则见文件头。

    轮次的切点都取在开爪帧**之后一帧**：闭爪帧本身仍属于它那一轮，
    于是 cycle_dist 在闭爪帧上正好读到 0（手已经到工具），语义就是「现在合爪」。
    """
    target = np.empty((n, 3), dtype=np.float64)
    idx = np.empty(n, dtype=np.float64)
    for k in range(5):
        a = 0 if k == 0 else release[k - 1] + 1
        r = release[k]
        target[a:r + 1] = tool_ref[k]      # 整个第 k 轮都盯着第 k 把
        idx[a:r + 1] = k + 1
    target[release[-1] + 1:] = tool_ref[4]  # idx=6 只表示「停」，目标维冻结在最后一把
    idx[release[-1] + 1:] = 6.0
    return target, idx


def main() -> None:
    # ================= 第一遍：求全局关键点常量 =================
    print("=" * 72)
    print("第一遍：扫描全部 episode，求全局关键点常量 + 校验周期数")
    print("=" * 72)

    all_tools, all_boxes, n_frames, all_cycles = [], [], [], []
    for ep in range(N_EP):
        s = read_state(ep)
        if s.shape[1] != N_IN:
            raise SystemExit(f"集{ep} 的 state 是 {s.shape[1]} 维，预期 {N_IN}")
        grasp, release = find_cycles(s)
        if len(grasp) != 5 or len(release) != 5:
            raise SystemExit(f"集{ep} 的周期数异常: 闭爪{len(grasp)} 开爪{len(release)}")
        ee = s[:, EE_SLICE]
        all_tools.append(ee[grasp])                 # (5,3)
        all_boxes.append(ee[release].mean(axis=0))  # (3,)
        n_frames.append(len(s))
        all_cycles.append((grasp, release))

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

    # 新维度的跨度自查：y 必须远大于抖动，cycle_dist 要覆盖到 0 附近
    print(f"\n  新维度跨度自查 (对照: 布局抖动 ±8mm, 12% 概率 ±40mm):")
    y_all = np.concatenate([TOOL_REF[:, 1], BOX_REF[1:2]])
    print(f"    goal_y     目标点 y 取值范围 [{y_all.min():+.4f}, {y_all.max():+.4f}]  "
          f"跨度 {(y_all.max() - y_all.min()) * 1000:.1f} mm")
    print(f"    相邻工具 y 差: "
          + ", ".join(f"{(TOOL_REF[i + 1, 1] - TOOL_REF[i, 1]) * 1000:+.1f}" for i in range(4)) + " mm")

    # ================= 第二遍：写数据 =================
    print()
    print("=" * 72)
    print(f"第二遍：写 {N_OUT} 维数据 -> {DST}")
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
    d18_lo, d18_hi = 1e9, -1e9
    cd_lo, cd_hi = 1e9, -1e9
    y_lo, y_hi = 1e9, -1e9
    cd_at_grasp_max = 0.0         # 5 个闭爪帧上 cycle_dist 应该≈0，记录最大偏离
    max_jump = 0.0                # 相邻帧最大跳变（和 v2 的 0.0728 对比）
    idx_hist = {k: 0 for k in range(1, 7)}

    for ep in range(N_EP):
        t = pq.read_table(os.path.join(SRC, f"data/chunk-000/episode_{ep:06d}.parquet"))
        state = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
        action = np.array(t["action"].to_pylist(), dtype=np.float32)
        if state.shape[1] != N_IN or action.shape[1] != N_IN:
            raise SystemExit(f"集{ep} state/action 维数 {state.shape[1]}/{action.shape[1]}，预期 {N_IN}")
        n = state.shape[0]
        ee = state[:, EE_SLICE].astype(np.float64)
        grasp, release = all_cycles[ep]

        # --- 第 18 维：沿用 v2 的 min_j‖ee − goals[j]‖，逐字不变 ---
        d18 = np.linalg.norm(ee[:, None, :] - GOALS[None, :, :], axis=2).min(axis=1)
        d18 = d18.astype(np.float32)
        d18_lo, d18_hi = min(d18_lo, float(d18.min())), max(d18_hi, float(d18.max()))

        # --- 新增 3 维：周期感知的目标点 ---
        target, idx = build_cycle_targets(grasp, release, n, TOOL_REF)
        goal_y = target[:, 1]
        cycle_dist = np.linalg.norm(ee - target, axis=1)
        cd_lo, cd_hi = min(cd_lo, float(cycle_dist.min())), max(cd_hi, float(cycle_dist.max()))
        y_lo, y_hi = min(y_lo, float(goal_y.min())), max(y_hi, float(goal_y.max()))
        for k in range(5):
            cd_at_grasp_max = max(cd_at_grasp_max, float(cycle_dist[grasp[k]]))
        max_jump = max(max_jump, float(np.abs(np.diff(cycle_dist)).max()))
        for k in range(1, 7):
            idx_hist[k] += int((idx == k).sum())

        new3 = np.stack([goal_y, cycle_dist, idx], axis=1).astype(np.float32)

        state22 = np.concatenate([state, d18[:, None], new3], axis=1)
        # action[f] == state[f+1]
        tail = np.concatenate([d18[:, None], new3], axis=1)
        tail_next = np.concatenate([tail[1:], tail[-1:]], axis=0)
        action22 = np.concatenate([action, tail_next], axis=1)

        assert state22.shape[1] == N_OUT and action22.shape[1] == N_OUT

        tbl = pa.table({
            "observation.state": pa.array([r.tolist() for r in state22], type=pa.list_(pa.float32())),
            "action": pa.array([r.tolist() for r in action22], type=pa.list_(pa.float32())),
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
        for key, arr in (("observation.state", state22), ("action", action22)):
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
        print(f"  集{ep:>3}: {n} 帧  idx 切换 {len(grasp)}+{len(release)} 次  "
              f"cycle_dist ∈ [{cycle_dist.min():.3f}, {cycle_dist.max():.3f}]", flush=True)

    # ================= meta =================
    info = json.load(open(os.path.join(SRC, "meta/info.json")))
    dim_names = info["features"]["observation.state"]["names"][0] + [DIM18] + NEW_DIMS
    info["total_episodes"] = N_EP
    info["total_frames"] = int(global_index)
    info["total_videos"] = N_EP * 2
    info["splits"] = {"train": f"0:{N_EP}"}
    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [N_OUT]
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
        "note": "右臂末端(base 系)的关键点。前 19 维沿用 v2：dim18 = min_j‖ee−goals[j]‖。"
                "新增 3 维(19..21)是**周期感知**的，推理端必须按同一套规则现算，"
                "见 target_rule。",
        "ee_slice": [7, 10],
        "grip_idx": GRIP_IDX,
        "grip_thresh": GRIP_THRESH,
        "tool_refs": TOOL_REF.tolist(),
        "box_ref": BOX_REF.tolist(),
        "goals": GOALS.tolist(),
        "dist_range_m": [d18_lo, d18_hi],
        "new_dims": NEW_DIMS,
        "new_dim_ranges": {
            "goal_y": [y_lo, y_hi],
            "cycle_dist": [cd_lo, cd_hi],
            "cycle_idx": [1.0, 6.0],
        },
        "target_rule": (
            "按夹爪方波切 5 轮(闭爪=grip_inner>thresh)。第 k 轮(0基, k=0..4):"
            " [上一轮开爪, 第k次闭爪) -> target=TOOL_REF[k];"
            " idx=k+1。第 4 次开爪之后 -> target 冻结在 TOOL_REF[4], idx=6"
            "（不切盒子：切了读数会从 0.655m 跳到 0）。"
            "dim19=target[1], dim20=‖ee-target‖, dim21=idx。"
        ),
        "index_advance": "推理端 idx 只能靠**数已经开过几次爪**推进(事件计数，不是时间)，"
                         "否则训练/推理错配。",
        "build": "v3-cycle (两遍扫描, 全局常量)",
    }
    json.dump(out, open(os.path.join(DST, "tool_refs.json"), "w"), indent=2)

    # ================= 汇总 =================
    tot = sum(idx_hist.values())
    print()
    print("=" * 72)
    print("完成")
    print("=" * 72)
    print(f"  目标: {DST}")
    print(f"  集数 {N_EP}, 总帧 {global_index}, 维数 {N_OUT}")
    print(f"  dim18 goal_dist  范围 [{d18_lo:.4f}, {d18_hi:.4f}] m   (应与 v2 完全一致)")
    print(f"  dim19 goal_y     范围 [{y_lo:+.4f}, {y_hi:+.4f}] m   跨度 {(y_hi - y_lo) * 1000:.1f} mm")
    print(f"  dim20 cycle_dist 范围 [{cd_lo:.4f}, {cd_hi:.4f}] m")
    print(f"  dim21 cycle_idx  范围 [1, 6]")
    print()
    print(f"  ** 闭爪帧归零校验 **  5 个闭爪帧上 cycle_dist 最大读数 "
          f"{cd_at_grasp_max * 1000:.2f} mm")
    print(f"     (不会精确到 0：TOOL_REF 是跨集均值，每集布局还差 ~1cm。"
          f"v2 同口径约 11 mm / p95 23 mm)")
    print(f"  相邻帧最大跳变 {max_jump:.4f} m   (v2 同口径 0.0728)")
    print()
    print(f"  idx 分布(帧数/占比):")
    for k in range(1, 7):
        print(f"    idx={k}: {idx_hist[k]:>7d}  {idx_hist[k] / tot * 100:5.1f}%")
    print(f"\n  关键点常量已写入 {DST}/tool_refs.json")


if __name__ == "__main__":
    main()
