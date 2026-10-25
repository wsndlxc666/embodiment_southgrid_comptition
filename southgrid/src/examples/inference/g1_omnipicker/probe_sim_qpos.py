"""直接摸仿真模型本身：关节名、qpos 地址、写入后读回、以及 site 的实时位姿。

前一个探针（probe_joint_map）结论：
  · official(`query_site_pos_and_quat_B`) == manual(`_live_ee_b` 手算) 到 0.00mm
    → runner 里的 `_live_ee_b` **没写错**，那条排查方向作废。
  · 但两者都给出「全 0 关节 → 0.536m 臂展」，而离线 FK 同关节是 1.026m。
  · 仿真模型 nq=110/njnt=56，离线模型 nq=49/njnt=43 —— 不是同一份模型！
  · 仿真里 base 世界位置 = [19.65, -23.95, 0.86]，离原点 20 米。

所以现在只查一件事：`set_joint_qpos` 之后，**仿真模型自己的 qpos 到底变成了什么**。
绕过所有封装，直接读 `env.gym._mjModel` / `env.gym._mjData`。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_sim_qpos.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402
from run_task3_tool_storage_scored import ROBOT_BASE_BODY, EE_SITE_R  # noqa: E402

Q_TGT = np.array([0.56471, 2.04204, -0.23143, 1.71851, -0.32236, -1.74, 0.68399])


def main() -> None:
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

    gm = env.gym
    m, d = gm._mjModel, gm._mjData
    print(f"仿真模型: nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody} nsite={m.nsite}")

    names_want = list(E.agent_conf.r_arm["joint_names"])
    print("\n右臂 7 关节在**仿真模型**里的名字 / id / qposadr / 类型 / 限位:")
    sim_names = [env.joint(n) for n in names_want]
    for raw, sn in zip(names_want, sim_names):
        jid = m.joint(sn).id
        print(f"  {raw:22s} -> {sn:44s} id={jid:3d} qposadr={m.jnt_qposadr[jid]:3d} "
              f"type={m.jnt_type[jid]} range=[{np.degrees(m.jnt_range[jid][0]):8.2f},"
              f"{np.degrees(m.jnt_range[jid][1]):8.2f}]°")

    print("\n站点/基座在仿真模型里:")
    for nm in (EE_SITE_R, ROBOT_BASE_BODY):
        try:
            sid = m.site(nm).id
            print(f"  site {nm} id={sid} body={m.body(m.body_id2name(m.site_bodyid[sid])).name}")
        except Exception as e:
            print(f"  {nm}: {e}")

    def read_q(adrs):
        return np.array([d.qpos[a] for a in adrs])

    def sim_ee():
        sid = m.site(EE_SITE_R).id
        bid = m.body(ROBOT_BASE_BODY).id
        p = np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])
        import mujoco
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        return Rb.T @ p

    adrs = [m.jnt_qposadr[m.joint(sn).id] for sn in sim_names]

    print(f"\n{'':<10}{'设定':>12}  {'写后立刻读回':>12}  {'mj_forward 后读回':>14}  "
          f"{'site 相对 base (手算自 _mjData)':>34}")
    print(f"{'初始':<10}{'—':>12}  {np.round(read_q(adrs),4).tolist()}"
          f"  {np.round(read_q(adrs),4).tolist()}  {np.round(sim_ee(),4).tolist()}")

    # 官方 API 写入
    env.set_joint_qpos({sn: np.float32(v) for sn, v in zip(sim_names, Q_TGT)})
    print(f"{'API 写入后':<10}{np.round(Q_TGT,4).tolist()}  {np.round(read_q(adrs),4).tolist()}")
    env.mj_forward()
    print(f"{'  再 forward':<10}{'':>12}  {'':>12}  {np.round(read_q(adrs),4).tolist()}  "
          f"{np.round(sim_ee(),4).tolist()}")

    # 直接写 _mjData，绕过所有封装
    for a, v in zip(adrs, Q_TGT):
        d.qpos[a] = v
    print(f"{'直写 _mjData':<10}{np.round(Q_TGT,4).tolist()}  {np.round(read_q(adrs),4).tolist()}")
    import mujoco
    mujoco.mj_forward(m, d)
    print(f"{'  再 forward':<10}{'':>12}  {'':>12}  {np.round(read_q(adrs),4).tolist()}  "
          f"{np.round(sim_ee(),4).tolist()}")

    print(f"\n离线 FK 对同一组关节的预测 = [0.1642, -0.2862, 0.4255]")


if __name__ == "__main__":
    main()
