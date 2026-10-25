"""同一个进程里跑三条 FK 路径，看它们是否一致 —— ik 解落不到目标就出在这里。

已实测（probe_align_repro.py）：把 start_ref 里的 ik_joints_r 原样写进 7 个关节，
直读 qpos 完全等于设定值，但
    · `query_site_pos_and_quat_B` 给的 base 系末端  = [-0.0385, 0.2806, 0.4581]
    · `mj_kinematics` 手算                          = [-0.0385, 0.2806, 0.4581]  （两者一致）
    · 目标（数据集帧 0）                            = [0.1585, -0.2792, 0.4356]
    差 593.83mm。
但 `probe_ik_sim.py` 当初对同一组关节报的是 0.16mm。两条 FK 必然有一处不同。

本脚本把三条路径并排：
  A 官方 query_site_pos_and_quat_B
  B 手算 Rb.T@(spos-bp)（不清 free joint）
  C probe_ik_sim.py 的做法：先把机器人 free joint 清成原点单位四元数，再 mj_kinematics
  D 同上但**不清** free joint
一次跑完，直接看哪条不一样。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_fk_paths.py
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

R_JOINTS = [f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)]
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
    print(f"目标 (数据集帧0 base 系) = {np.round(p_t, 5).tolist()}")
    print(f"待验证 ik_joints_r       = {np.round(q_ik, 6).tolist()}\n")

    m, d = env.gym._mjModel, env.gym._mjData
    jids = [m.joint(env.joint(n)).id for n in R_JOINTS]
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    sid = m.site(EE_SITE_R).id
    bid = m.body(ROBOT_BASE_BODY).id
    print(f"qadr(7 个右臂关节的 qpos 地址) = {qadr}   互不相同? {len(set(qadr))==7}")

    free_jid = m.joint(env.joint("free_joint")).id
    free_adr = int(m.jnt_qposadr[free_jid])
    free_bid = int(m.jnt_bodyid[free_jid])
    print(f"机器人 free joint: id={free_jid} adr={free_adr} "
          f"挂在 body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, free_bid)}")
    print(f"base body {ROBOT_BASE_BODY} id={bid}  parentid="
          f"{m.body_parentid[bid]}({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[bid]))})")
    print(f"  它到 root 之间还有 {m.body_jntnum[bid]} 个关节\n")

    q0 = np.array(d.qpos).copy()

    def official():
        q = env.query_site_pos_and_quat_B([EE_SITE_R], [ROBOT_BASE_BODY])[EE_SITE_R]
        return np.asarray(q["xpos"], dtype=np.float64).reshape(3)

    def manual(zero_free: bool):
        d.qpos[:] = q0
        if zero_free:
            d.qpos[free_adr:free_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[qadr] = q_ik
        mujoco.mj_kinematics(m, d)
        p = np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        return Rb.T @ p

    # teleport 走官方接口，再分别量
    env.set_joint_qpos({env.joint(n): np.float32(v) for n, v in zip(R_JOINTS, q_ik)})
    env.mj_forward()
    a = official()
    b = manual(False)
    c = manual(True)
    env.mj_forward()
    dd = manual(False)

    rows = [
        ("A 官方 query_site_pos_and_quat_B", a),
        ("B 手算 Rb.T@(spos-bp)，不清 free", b),
        ("C 手算，把 free joint 清成原点(probe_ik_sim 的做法)", c),
        ("D 手算，再 mj_forward 一次后不清 free", dd),
    ]
    for name, p in rows:
        print(f"  {name:52s} = {np.round(p, 5).tolist()}   距目标 {np.linalg.norm(p - p_t)*1000:8.2f} mm")

    # 现场做一次最小 IK 复核：用 C 这条路自己从 q_ik 出发做几步高斯牛顿，看能不能到 0
    print("\n[复核] 用 C 这条路，从 q_ik 出发再迭代 60 步看能否收敛到目标：")
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    q = q_ik.copy()
    h = 1e-6

    def fkC(qq):
        d.qpos[:] = q0
        d.qpos[free_adr:free_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[qadr] = qq
        mujoco.mj_kinematics(m, d)
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        return Rb.T @ (np.array(d.site_xpos[sid]) - np.array(d.xpos[bid]))

    def err(qq):
        return p_t - fkC(qq)

    e = err(q)
    print(f"  起点残差 {np.linalg.norm(e)*1000:.2f} mm")
    for it in range(60):
        J = np.zeros((3, 7))
        for i in range(7):
            qp = q.copy()
            qp[i] = q[i] + h if (q[i] + h) <= hi[i] else q[i] - h
            J[:, i] = (err(qp) - e) / (qp[i] - q[i])
        try:
            dq = np.linalg.lstsq(J, e, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        q = np.clip(q + dq, lo, hi)
        e = err(q)
        if np.linalg.norm(e) < 1e-5:
            break
    print(f"  迭代后残差 {np.linalg.norm(e)*1000:.2f} mm")
    print(f"  解出的关节(度) = {np.round(np.degrees(q), 2).tolist()}")


if __name__ == "__main__":
    main()
