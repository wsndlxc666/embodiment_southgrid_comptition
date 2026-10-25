"""初始化位姿探针 —— 确认 ④ 递给模型的"第一帧"到底是哪个位姿。

背景（2026-09-17 排查）：训练集 100 集的第一帧 state 逐维完全相同（std=0），
说明采集器每次录第 0 帧时机器人都停在一个确定的位姿上。而 ④ 那次演练里
i=0 的右臂位形在训练集里**找不到任何一帧能匹配**（最近帧距离 0.289），
`cycle_dist` 更是 0.145（训练帧 0 = 0.510）。位姿不对 → 模型从第一帧起就在
分布外 → 它 0.5 秒就把爪子合了一下，把 cycle_idx 推过了第 0 轮。

这个脚本只做一件事：把 ④ 的初始化序列原样走一遍，在四个时刻读 state，
分别和"训练帧 0"、"上次运行 i=0"比对，看偏差出在哪一步。

用法（orcalab 绝对路径 python）：
    .../bin/python probe_init_state.py --tool_refs tool_refs_d22.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

NAMES = ["lpx", "lpy", "lpz", "lqx", "lqy", "lqz", "lqw",
         "rpx", "rpy", "rpz", "rqx", "rqy", "rqz", "rqw",
         "lg_in", "lg_out", "rg_in", "rg_out",
         "goal_dist", "goal_y", "cycle_dist", "cycle_idx"]


def _train_frame0(dim: int = 22) -> np.ndarray | None:
    """训练集第一帧的 state（100 集完全相同，取第 1 集）。"""
    import pandas as pd
    for d in ("g1_omnipicker_clear5_d22_100ep", "g1_omnipicker_clear5_d19_goal_100ep",
              "g1_omnipicker_clear5_d19_100ep", "g1_omnipicker_clear5_train"):
        p = f"/home/dgl/lxc/openpi/data/{d}"
        fs = sorted(glob.glob(p + "/data/**/*.parquet", recursive=True))
        if fs:
            st = np.stack(pd.read_parquet(fs[0])["observation.state"].values)
            print(f"[ref] 训练帧0 取自 {d} ({len(st)} 帧, state {st.shape[1]} 维)")
            return st[0][:dim]
    print("[ref] 找不到训练集 parquet")
    return None


def _run_i0(path: str) -> np.ndarray | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return np.asarray(json.loads(f.readline())["state"], dtype=np.float64)
    except Exception as e:
        print(f"[ref] 读不到上次运行转储 {path}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--prev_dump", type=str,
                    default="/home/dgl/lxc/.runlogs/d22_no_score_chunks.jsonl")
    ap.add_argument("--settle_steps", type=int, default=180,
                    help="采集器 _settle 的步数，用于对比'走够仿真步'之后会停在哪")
    args = ap.parse_args()

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)

    tool_refs = cycle_refs = None
    with open(args.tool_refs, "r", encoding="utf-8") as f:
        _tr = json.load(f)
    if "goals" in _tr:
        tool_refs = np.asarray(_tr["goals"], dtype=np.float32).reshape(-1, 3)
        ref_name = "goal_dist"
    else:
        tool_refs = np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
        ref_name = "tool_dist"
    if _tr.get("new_dims"):
        cycle_refs = np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(
        dataset_path="/tmp/_probe_scratch", tool_refs=tool_refs, ref_name=ref_name,
        cycle_refs=cycle_refs,
    )
    storage.reset_cycle()
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=E.ENTRY_POINT,
        default_joint_values={},
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE

    def snap(tag: str) -> np.ndarray:
        storage.reset_cycle()          # 让 cycle_idx 每次都从 1 起，四帧才可比
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        print(f"\n--- {tag} ---  state={s.shape[0]} 维")
        for k in range(min(14, s.shape[0])):
            print(f"    [{k:2d}] {NAMES[k]:<10s} {s[k]:+.4f}")
        for k in range(14, s.shape[0]):
            print(f"    [{k:2d}] {NAMES[k]:<10s} {s[k]:+.4f}")
        return s

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    S0 = snap("S0  env.reset()+update_scene() 之后（teleport 之前 = 场景原始位姿）")

    # ---- ④ 的初始化序列，逐行照抄 ----
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()
    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    for c in manager.controllers:
        c.reset()
    env.render()
    time.sleep(0.05)
    S1 = snap("S1  ④ 的 teleport + mj_forward + set_init_ctrl + c.reset() 之后")

    # ---- 参考实现多做的那步：用首帧观测重置末端目标 ----
    try:
        _a = E.build_initial_action_from_state(S1[:18])
        _ap = E.action_dict_for_apply(_a)
        print(f"\n[ref] 参考实现会把末端目标设成 {_a}（④ 没做这一步，只用 c.reset()）")
    except Exception as e:
        print(f"[ref] build_initial_action_from_state 失败: {e}")

    # ---- 采集器的 _settle：走够仿真步之后停在哪 ----
    if args.settle_steps > 0:
        print(f"\n[settle] 按采集器 _settle 的写法走 {args.settle_steps} 步仿真 …")
        try:
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            for _ in range(args.settle_steps):
                env.step(manager.ctrl)
        except Exception as e:
            print(f"[settle] 异常: {e}")
        S2 = snap(f"S2  settle {args.settle_steps} 步之后")
    else:
        S2 = None

    ref_tr = _train_frame0()
    ref_run = _run_i0(args.prev_dump)

    print("\n" + "=" * 78)
    print("比对：右臂位置(7:10) / 右臂四元数(10:14) / cycle_dist(20)")
    print(f"{'来源':<44s} {'右臂位置':>26s} {'cyc_d':>8s}")
    for name, s in [("S0 teleport之前(场景原始)", S0), ("S1 ④初始化之后", S1),
                    ("S2 settle之后", S2), ("训练帧0(标准答案)", ref_tr),
                    ("上次运行 i=0", ref_run)]:
        if s is None:
            continue
        rp = "(" + ",".join(f"{v:+.3f}" for v in s[7:10]) + ")"
        cd = f"{s[20]:.4f}" if s.shape[0] > 20 else "n/a"
        print(f"{name:<44s} {rp:>26s} {cd:>8s}")
    print("=" * 78)
    if ref_tr is not None:
        for name, s in [("S0", S0), ("S1", S1), ("S2", S2), ("上次运行 i=0", ref_run)]:
            if s is None:
                continue
            d7 = float(np.linalg.norm(s[7:14] - ref_tr[7:14]))
            d0 = float(np.linalg.norm(s[0:7] - ref_tr[0:7]))
            print(f"  {name:<14s} 距训练帧0: 右臂(7:14) {d7:.4f}   左臂(0:7) {d0:.4f}"
                  f"   {'← 右臂匹配' if d7 < 0.02 else ''}")
    print("\n[结论] S1 若等于'训练帧0' → teleport 生效、④ 初始化没问题；"
          "若 S1 ≈ S0 而 S2 ≈ 训练帧0 → 缺的是'走够仿真步'。")


if __name__ == "__main__":
    main()
