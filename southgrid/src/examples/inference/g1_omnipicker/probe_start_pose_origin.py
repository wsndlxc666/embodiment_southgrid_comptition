"""训练帧 0 那个位姿到底是**怎么来的**？—— 逐个候选关节组去仿真实测。

2026-09-17 查采集器 g1_omnipicker_collect_clear5_lerobot.py 得到的新事实：
  · 采集器的初始化是 `_settle(env, manager, 180)`：`set_init_ctrl()` + `env.step(mgr.ctrl)`
    跑 180 个 env 步，而右臂的 `motors_init_ctrl` 是 **全 0**（不是 neutral！）。
  · 之后 `env.set_default_joint_values(...)` + `env.mj_forward()`。
  · 所以训练帧 0 极可能是**关节全 0** 摆出来的那个位姿，而不是 conf neutral。

如果成立，那 ④ 只要在开局把右臂关节直接置成和采集器一样（也就是 0 或 neutral，
纯配置、零训练），起始状态就和数据采集完全一致 —— 比现在的"伺服到帧 0 的笛卡尔位姿"
干净得多（伺服只能到位置 0.4mm/姿态差 131.5°，而直接摆关节是精确的）。

本脚本依次把右臂关节置成若干候选组，读 state，和数据集帧 0 比位置/姿态误差，
最后按误差排序，给出"最像数据采集起始状态"的那个候选。

用法：/home/dgl/miniconda3/envs/orcalab/bin/python probe_start_pose_origin.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start_ref", type=str, default=os.path.join(base_dir, "start_ref_d22.json"))
    args = ap.parse_args()

    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    from yaml import Loader, load

    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt = st[0].astype(np.float64)
    p_t, q_t = tgt[7:10], tgt[10:14]
    R_t = R.from_quat(q_t)
    print(f"[目标] 数据集帧0  state = {np.round(tgt, 4).tolist()}")
    print(f"        右臂末端 pos = {np.round(p_t, 5).tolist()}")
    print(f"        右臂末端 quat= {np.round(q_t, 5).tolist()}")
    print(f"        夹爪 norm    = {np.round(tgt[16:18], 5).tolist()}")
    print(f"        (跨 100 集极差请另测；本脚本只比 ep{args.episode})")

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr=args.orcagym_addr)
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]
    l_idx = [env.joint(n) for n in E.agent_conf.l_arm["joint_names"]]
    R_neutral = np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64)

    def put(q7):
        d = {j: np.float32(0.0) for j in l_idx}
        d.update({j: np.float32(v) for j, v in zip(jidx, q7)})
        env.set_joint_qpos(d)
        env.mj_forward()

    def read():
        s = np.asarray(storage.build_state(storage.obs_callback(env)),
                       dtype=np.float64).reshape(-1)
        dp = float(np.linalg.norm(s[7:10] - p_t))
        da = float(np.degrees((R_t * R.from_quat(s[10:14]).inv()).magnitude()))
        return dp, da, s

    cands: list[tuple[str, np.ndarray]] = [
        ("右臂关节全 0（= 采集器 motors_init_ctrl）", np.zeros(7)),
        ("右臂 conf neutral [0,0,0,1.5708,0,0,0]", R_neutral.copy()),
        ("左臂全0 + 右臂全0 时的默认（set_default）", None),  # 占位，见下
    ]

    # 先测"什么都不动"的默认状态（刚 set_default_joint_values + mj_forward 之后）
    dp, da, s_def = read()
    print(f"\n[候选 0] 默认状态（set_default_joint_values + mj_forward，不动关节）")
    print(f"          pos = {np.round(s_def[7:10], 5).tolist()}  位置误差 {dp*1000:7.2f} mm"
          f"  姿态误差 {da:7.2f}°  夹爪 {s_def[16]:.4f}")

    rows = []
    for name, q in cands:
        if q is None:
            continue
        put(q)
        dp, da, s = read()
        rows.append((dp, da, name, s, q))
        print(f"\n[候选] {name}")
        print(f"          pos = {np.round(s[7:10], 5).tolist()}  位置误差 {dp*1000:7.2f} mm"
              f"  姿态误差 {da:7.2f}°")
        print(f"          全量 state 最大维度偏差 {np.abs(s - tgt[:s.size]).max():.4f}")

    # 采集器实际流程的复刻：zeros 跑 180 步 OSC 之后再读（settle 的终点，而非瞬间值）
    print(f"\n[复刻采集器] reset → set_default_joint_values → 用 init ctrl(全0) 跑 180 个 env 步 …")
    env.reset(); time.sleep(0.05)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()
    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    l_arm = E.create_arm(env, E.agent_conf.l_arm)
    r_arm = E.create_arm(env, E.agent_conf.r_arm)
    for _a in (l_arm, r_arm):
        _a.controller.kp = np.ones(6, dtype=np.float64) * 220.0
        _a.controller.kd = 2.0 * np.sqrt(_a.controller.kp)
    l_grip = E.create_gripper(env, E.agent_conf.gripper_l)
    r_grip = E.create_gripper(env, E.agent_conf.gripper_r)
    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)
    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    t0 = time.time()
    for _ in range(180):
        env.step(manager.ctrl)
    dp, da, s_settle = read()
    rows.append((dp, da, "采集器复刻：init ctrl 全0 跑 180 步后的稳态", s_settle, None))
    print(f"          用时 {time.time()-t0:.1f}s")
    print(f"          pos = {np.round(s_settle[7:10], 5).tolist()}  位置误差 {dp*1000:7.2f} mm"
          f"  姿态误差 {da:7.2f}°  夹爪 {s_settle[16]:.4f}")
    print(f"          全量 state 最大维度偏差 {np.abs(s_settle - tgt).max():.4f}")

    rows.sort(key=lambda r: r[0])
    print(f"\n===== 按位置误差排序 =====")
    for dp, da, name, _s, _q in rows:
        print(f"  {dp*1000:9.2f} mm  {da:8.2f}°   {name}")

    best = rows[0]
    if best[0] < 0.010 and best[1] < 10.0:
        print(f"\n>>> 命中：`{best[2]}` 就是数据采集的起始状态。"
              f"④ 开局照它摆即可（纯配置，不需要伺服、不需要训练）。")
    else:
        print(f"\n>>> 没有候选能精确复现帧 0（最好 {best[0]*1000:.1f}mm / {best[1]:.1f}°）。"
              f"\n    说明帧 0 是脚本轨迹起步后的样子，见采集器 _build_tool_segments。")


if __name__ == "__main__":
    main()
