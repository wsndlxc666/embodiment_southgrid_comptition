"""为什么 `--start_align ik` 有时 0.7mm、有时 284mm？以及腕相机到底挂在哪？

2026-09-17 两次演练同一个脚本、同一个场景、同一组 ik 关节：
    ik3（前一次结束在"手臂下垂"姿态）→ teleport 后残差 0.72mm / 0.02°
    ik4（前一次结束在"卡住的抬臂"姿态）→ teleport 后残差 284.20mm / 103.22°
场景探针打印的工具/机器人世界坐标两次完全一致（8 位小数），所以不是场景随机化。

本脚本回答三件事：
  1. `E.agent_conf.r_arm["joint_names"]` 到底是什么名字、什么**顺序**，
     与 probe_ik_sim.py 用的 idx61..67 是否一致（顺序错了就等于把 7 个值写乱了）。
  2. 直接 teleport 那 7 个关节后，直读 `_mjData.qpos` 与 `query_site_pos_and_quat_B`
     给的 base 系末端，和 IK 目标差多少 —— 在**当前**服务端状态下复现。
  3. 腕相机图像是否随手臂姿态变化（挂臂 vs 固定机位）。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_align_repro.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402
from run_task3_tool_storage_scored import ROBOT_BASE_BODY, EE_SITE_R  # noqa: E402

SR = os.path.join(base_dir, "start_ref_d22.json")


def main() -> None:
    import mujoco
    from yaml import Loader, load

    cfg = os.path.abspath(os.path.join(base_dir, "../../dataCollection/common/example.yaml"))
    with open(cfg, "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    sm = E.SceneManager("localhost:50051", config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=sm, frame_skip=5, orcagym_addr="localhost:50051")
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset()
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()

    with open(SR, "r", encoding="utf-8") as f:
        sr = json.load(f)
    q_ik = np.asarray(sr["ik_joints_r"], dtype=np.float64).reshape(7)
    p_t = np.asarray(sr["right_ee_pos_b"], dtype=np.float64).reshape(3)
    q_t = np.asarray(sr["right_ee_quat_b"], dtype=np.float64).reshape(4)

    # ---- 1. 关节名与顺序 ----
    names = list(E.agent_conf.r_arm["joint_names"])
    print("=" * 90)
    print('1) E.agent_conf.r_arm["joint_names"] =')
    for i, n in enumerate(names):
        print(f"     [{i}] {n}")
    print(f"   IK 用的顺序是 idx61..67 = {[f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)]}")
    print(f"   两者一致？ {names == [f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)]}")

    # ---- 2. teleport 后直读 qpos 与 base 系末端 ----
    m, d = env.gym._mjModel, env.gym._mjData
    jids = [m.joint(env.joint(n)).id for n in names]
    qadr = [m.jnt_qposadr[j] for j in jids]

    def live_ee_b():
        q = env.query_site_pos_and_quat_B([EE_SITE_R], [ROBOT_BASE_BODY])[EE_SITE_R]
        pos = np.asarray(q["xpos"], dtype=np.float64).reshape(3)
        quat = np.asarray(q["xquat"], dtype=np.float64).reshape(4)[[1, 2, 3, 0]]
        return pos, quat

    print("\n" + "=" * 90)
    print("2) teleport 前后（当前服务端状态）")
    p0, q0_ = live_ee_b()
    print(f"   写之前  qpos = {np.round(np.array(d.qpos)[qadr], 4).tolist()}")
    print(f"   写之前  末端(base) = {np.round(p0, 4).tolist()}  距目标 {np.linalg.norm(p0-p_t)*1000:.2f} mm")
    env.set_joint_qpos({env.joint(n): np.float32(v) for n, v in zip(names, q_ik)})
    env.mj_forward()
    p1, q1 = live_ee_b()
    print(f"   写之后  qpos = {np.round(np.array(d.qpos)[qadr], 4).tolist()}")
    print(f"   目标    qpos = {np.round(q_ik, 4).tolist()}")
    print(f"   写之后  末端(base) = {np.round(p1, 4).tolist()}  距目标 {np.linalg.norm(p1-p_t)*1000:.2f} mm")
    from scipy.spatial.transform import Rotation as R
    ang = np.degrees(np.linalg.norm((R.from_quat(q_t).inv() * R.from_quat(q1)).as_rotvec()))
    print(f"   姿态误差 {ang:.2f}°")

    # 用「重算 FK」的办法独立核对：把 qpos 直接写进模型算 site
    sid = m.site(EE_SITE_R).id
    bid = m.body(ROBOT_BASE_BODY).id
    mujoco.mj_kinematics(m, d)
    pf = np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])
    Rb = np.array(d.xmat[bid]).reshape(3, 3)
    print(f"   mj_kinematics 手算(base) = {np.round(Rb.T @ pf, 4).tolist()}"
          f"  距目标 {np.linalg.norm(Rb.T @ pf - p_t)*1000:.2f} mm")

    # ---- 3. 腕相机是否随手臂动 ----
    print("\n" + "=" * 90)
    print("3) 相机对象在哪儿")
    try:
        print(f"   env 上含 cam 的属性: {[a for a in dir(env) if 'cam' in a.lower()]}")
        print(f"   manager 上含 cam 的属性: {[a for a in dir(manager) if 'cam' in a.lower()]}")
        gym = env.gym
        print(f"   gym 上含 cam 的属性: {[a for a in dir(gym) if 'cam' in a.lower()]}")
    except Exception as exc:
        print(f"   相机检查失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
