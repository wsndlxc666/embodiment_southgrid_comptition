"""不用 IK：让 ④ 自己的 OSC 控制器把右臂**伺服**到 clear5 帧 0 的位姿。

为什么换这个方法：
  · 之前用随机起点做 IK，7 个未知量解 3~6 个约束，起点稍差就落进劣质盆地 ——
    位置能到 2.91mm 但姿态差 135°，再换种子又从零开始，结果不可复现（probe_ik_final
    直接退化到 24mm/168°）。**不该由我来解 IK。**
  · OSC 控制器本来就是个笛卡尔阻抗控制器，喂它一个目标位姿它自己就会算力矩过去。
    对冗余臂，它解出的那组关节角是"离当前姿态最近的解" —— 良定义、可复现，
    而且**用的是评测时真正在用的那套控制器**。

测试内容：从 ④ 的当前初始姿态（左臂全零 / 右臂 conf neutral）出发，
把 clear5_d22 帧 0 的右臂位姿当目标下发，看能不能收敛、收敛到哪、关节角是多少、
是否在限位内。收敛了就顺带把关节角打印出来（将来可选硬编码，但伺服本身更稳）。

用法：.../bin/python probe_servo_start.py
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

LIMITS_DEG = [(-179.908752441, 179.908752441), (-119.748168945, 119.748168945),
              (-177.616912842, 177.616912842), (0.0, 100.267608643),
              (-177.616912842, 177.616912842), (-99.694656372, 99.694656372),
              (-177.616912842, 177.616912842)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--kp", type=float, default=150.0)
    ap.add_argument("--max_s", type=float, default=25.0)
    ap.add_argument("--pos_only", action="store_true",
                    help="只伺服位置：姿态目标每步取当前实际姿态，不去跟姿态较劲。\n                         用来分辨 50mm 残差到底是位置够不着，还是被姿态约束拖住的。")
    ap.add_argument("--hold_s", type=float, default=6.0, help="收敛后再保持几秒，看会不会被拖走")
    args = ap.parse_args()

    import pandas as pd
    from scipy.spatial.transform import Rotation as R

    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt = st[args.frame].astype(np.float64).copy()
    print(f"[目标] clear5 第{args.episode}集 帧{args.frame} state({tgt.size}) = {np.round(tgt, 4).tolist()}")

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)
    with open(args.tool_refs, "r", encoding="utf-8") as f:
        _tr = json.load(f)
    tool_refs = np.asarray(_tr.get("goals", _tr["tool_refs"]), dtype=np.float32).reshape(-1, 3)
    ref_name = "goal_dist" if "goals" in _tr else "tool_dist"
    cycle_refs = (np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
                  if _tr.get("new_dims") else None)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(
        dataset_path="/tmp/_probe_scratch", tool_refs=tool_refs, ref_name=ref_name,
        cycle_refs=cycle_refs)
    storage.reset_cycle()
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr=args.orcagym_addr)
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE

    # ---- 与 ④ 完全一致的初始化 ----
    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()
    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    for c in manager.controllers:
        c.reset()
    env.render(); time.sleep(0.05)

    l_arm = E.create_arm(env, E.agent_conf.l_arm)
    r_arm = E.create_arm(env, E.agent_conf.r_arm)
    for _a in (l_arm, r_arm):
        _a.controller.kp = np.ones(6, dtype=np.float64) * args.kp
        _a.controller.kd = 2.0 * np.sqrt(_a.controller.kp)
    l_grip = E.create_gripper(env, E.agent_conf.gripper_l)
    r_grip = E.create_gripper(env, E.agent_conf.gripper_r)
    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)

    s_start = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
    print(f"[起点] 当前 state({s_start.size}) = {np.round(s_start, 4).tolist()}")

    # 目标动作 = 训练帧 0 的 state 前 14 维（**state 与 action 的前 14 维布局相同**），
    # 夹爪也取训练帧 0 的值，这样送过去的就是一个完整的"训练起始状态"。
    tgt18 = np.zeros(18, dtype=np.float32)
    tgt18[:14] = tgt[:14]
    tgt18[14:18] = tgt[14:18]
    tgt_apply = E.action_dict_for_apply(E.parse_policy_action(tgt18))

    device = E.EEFDevice(
        l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
        lock_left_arm=True, grasp_integral=False, **tgt_apply,
    )
    manager.set_device(device)
    device.set_left_hold(tgt_apply["l_pos_b"], tgt_apply["l_quat_b"], tgt_apply["l_grip_ctrl"])

    p_t, q_t = tgt[7:10], tgt[10:14]
    R_t = R.from_quat(q_t)

    def read():
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        dp = float(np.linalg.norm(s[7:10] - p_t))
        try:
            da = float(np.degrees((R_t * R.from_quat(s[10:14]).inv()).magnitude()))
        except Exception:
            da = float("nan")
        return dp, da, s

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]
    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    def joints():
        d = env.query_joint_qpos(jidx)
        return np.array([float(np.asarray(d[j]).reshape(-1)[0]) for j in jidx], dtype=np.float64)

    print(f"\n[伺服] 把右臂位姿目标设成训练帧 0，最多跑 {args.max_s:.0f}s …")
    t0 = time.time()
    last = 0.0
    while (time.time() - t0) < args.max_s:
        if args.pos_only:
            _cur = np.asarray(storage.build_state(storage.obs_callback(env)),
                              dtype=np.float64).reshape(-1)
            _a = dict(tgt_apply)
            _a["r_quat_b"] = np.asarray(_cur[10:14], dtype=np.float32)
            device.set_target(**_a)
        else:
            device.set_target(**tgt_apply)
        act = manager.run_controllers()
        env.step(act)
        env.render()
        el = time.time() - t0
        if el - last >= 1.0:
            last = el
            dp, da, _ = read()
            q = joints()
            print(f"    t={el:5.1f}s  位置误差 {dp*1000:7.2f} mm  姿态误差 {da:7.2f}°  "
                  f"j = {np.round(q, 3).tolist()}", flush=True)
            if dp < 0.003 and da < 3.0:
                break

    dp, da, s_fin = read()
    q = joints()
    inlim = bool(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))

    print(f"\n===== 伺服结果（{time.time()-t0:.1f}s）=====")
    print(f"  目标 pos  = {np.round(p_t, 4).tolist()}")
    print(f"  到达 pos  = {np.round(s_fin[7:10], 4).tolist()}   位置误差 {dp*1000:.2f} mm")
    print(f"  目标 quat = {np.round(q_t, 4).tolist()}")
    print(f"  到达 quat = {np.round(s_fin[10:14], 4).tolist()}   姿态误差 {da:.2f}°")
    print(f"  右臂关节  = {np.round(q, 4).tolist()}")
    print(f"  限位内    = {inlim}")

    # ---- 交叉验证：把这些关节角直接 teleport 进关节，再看 state 对不对 ----
    env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
    env.mj_forward()
    storage.reset_cycle()
    s_tp = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
    dp2 = float(np.linalg.norm(s_tp[7:10] - p_t))
    da2 = float(np.degrees((R_t * R.from_quat(s_tp[10:14]).inv()).magnitude()))
    print(f"\n  [teleport 交叉验证] 位置 {dp2*1000:.2f} mm  姿态 {da2:.2f}°  "
          f"（与上面一致就说明 servo 到的是个真解，不是控制器没到位）")

    if dp < 0.005 and da < 5.0 and inlim:
        print(f"\n>>> 可用。把下面这两行加进 ④ 的初始化（在 device 建好、起始 ctrl 下发之后）：")
        print(f"    E.servo_right_to(manager, env, device, {np.round(tgt[:14],6).tolist()})")
        print(f"    等价关节角 = {np.round(q, 4).tolist()}")
    else:
        print(f"\n>>> 没收敛到可接受精度。位置 {dp*1000:.1f}mm / 姿态 {da:.1f}°。"
              f"\n    若位置也进不去，说明该位姿今天的模型摆不出来（比 412mm 那个变体假设更要紧的结论）。")


if __name__ == "__main__":
    main()
