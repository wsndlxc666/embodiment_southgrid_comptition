"""最后一个坐标系假设：right EE 当时是不是相对**另一个 body** 表达的？

clear5 帧 0 的右臂 = (0.1585, -0.2792, 0.4356, ...)，而今天 conf neutral 的右臂相对
body_link1 = (0.6822, -0.0802, 0.2355, ...)。差 594mm，且我用 69 种关节变体 + 随机起点
IK + OSC 伺服都够不着 —— 那说明"关节角不同"这个方向可能整个是错的。

更便宜的解释：**base_body 变了**。state 是 `query_site_pos_and_quat_B(ee_sites, [body(base_body)])`，
base_body 换一个，同一个关节角就给出完全不同的数字。所以这里把右臂末端相对**每一个候选
body** 都算一遍，看哪个能给出 (0.1585, -0.2792, 0.4356)。

注意：旧数据集 g1_omnipicker_tool 帧 0 的右臂 (0.6822,-0.0802,0.2352) 今天能被
body_link1 复现到 2e-4 —— 所以**旧数据**用的是 body_link1。问题是 clear5 用的是不是同一个。

用法：.../bin/python probe_base_body.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

TGT = np.array([0.1585, -0.2792, 0.4356, -0.6987, 0.5574, 0.3993, 0.2043])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    args = ap.parse_args()

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
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

    # 摆到 clear5 采集时的左右臂关节角（左=全零，右=conf neutral；这是已知的）
    lq = list(E.agent_conf.l_arm["neutral_joint_values"])
    rq = list(E.agent_conf.r_arm["neutral_joint_values"])
    d = {}
    for n, v in zip(E.agent_conf.l_arm["joint_names"], [0.0] * 7):
        d[env.joint(n)] = np.float32(v)
    for n, v in zip(E.agent_conf.r_arm["joint_names"], rq):
        d[env.joint(n)] = np.float32(v)
    env.set_joint_qpos(d)
    env.mj_forward()

    site = "g1_omnipicker_ee_center_site_r"
    print(f"目标（clear5 帧0 右臂）= {np.round(TGT, 4).tolist()}\n")

    # 候选 base body：机器人链上所有 body + 一些常见别名
    cands = []
    for pre in ("g1_omnipicker_", ""):
        for b in ("body_link1", "body_link2", "body_link3", "head_link", "arm_r_link1",
                  "arm_r_link2", "arm_r_link3", "arm_r_link4", "arm_r_link5",
                  "arm_r_link6", "arm_r_link7", "arm_r_link8", "torso_link",
                  "pelvis", "base_link", "ActorManipulator_Anchor",
                  "ORCA_MANIPULATOR_a3f5e2d1-7b8c-4f2a-9e6d-1c2b3a4f5d6e_Anchor"):
            cands.append(pre + b)
    cands = ["world"] + cands

    print(f"{'base body':<62s} {'右臂末端 (pos / quat)':>58s} {'位置差':>10s}")
    best = None
    for b in cands:
        try:
            bn = env.body(b) if b != "world" else "world"
            q = env.query_site_pos_and_quat_B([site], [bn])
            xpos = np.asarray(q[site]["xpos"], dtype=np.float64).reshape(3)
            xq = np.asarray(q[site]["xquat"], dtype=np.float64).reshape(4)[[1, 2, 3, 0]]  # wxyz->xyzw
        except Exception:
            continue
        dp = float(np.linalg.norm(xpos - TGT[:3]))
        dq = float(np.linalg.norm(xq - TGT[3:]))
        if best is None or dp < best[0]:
            best = (dp, b, xpos, xq)
        flag = "  <<<" if dp < 0.02 else ""
        print(f"{b:<62s} {str(np.round(np.concatenate([xpos, xq]), 4).tolist()):>58s} "
              f"{dp*1000:>7.1f} mm{flag}")

    dp, b, xpos, xq = best
    print(f"\n最接近的 base body = {b}  位置 {dp*1000:.1f} mm  四元数差 {np.linalg.norm(xq-TGT[3:]):.4f}")
    if dp < 0.02:
        print("  ⇒ 命中！clear5 采集时 state 用的 base_body 就是这个 —— 关节角其实没变。")
    else:
        print("  ⇒ 都不是。base_body 不是原因，clear5 帧 0 确实对应另一组关节角。")


if __name__ == "__main__":
    main()
