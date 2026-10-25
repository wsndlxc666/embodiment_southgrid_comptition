"""逐项对比「仿真模型」与「离线 MJCF」的右臂链参数，找出 FK 不一致的确切位置。

已排除：写回失败（直读 _mjData.qpos = 设定值）、链结构不同（两者都是
body_link1 → body_link2 → arm_r_link1..7 → ee site，中间恰好 7 个关节）、
`_live_ee_b` 手算错误（official 与 manual 差 0.00mm）。

剩下的只有一种可能：**两边的链参数不同** —— body_pos / body_quat / jnt_axis /
jnt_pos / jnt_range 里有一项对不上，于是同一组关节角给出不同末端位姿。

本脚本把两边的这几项并排打印并标出差异行。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_sim_vs_offline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

SITE = "g1_omnipicker_ee_center_site_r"
BASE = "g1_omnipicker_body_link1"
PREFIX = "g1_omnipicker_"
XML = os.path.join(base_dir, "..", "..", "..", "model", "G1_omnipicker", "g1_omnipicker.xml")


def dump(m, strip: bool):
    """返回 {body名: (pos, quat)} 与 {关节名: (axis, pos, range)}，键名去掉前缀。"""
    import mujoco
    bodies, joints = {}, {}
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if n is None:
            continue
        k = n[len(PREFIX):] if (strip and n.startswith(PREFIX)) else n
        bodies[k] = (np.array(m.body_pos[i]), np.array(m.body_quat[i]))
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n is None:
            continue
        k = n[len(PREFIX):] if (strip and n.startswith(PREFIX)) else n
        joints[k] = (np.array(m.jnt_axis[j]), np.array(m.jnt_pos[j]),
                     np.array(m.jnt_range[j]))
    return bodies, joints


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

    m_sim = env.gym._mjModel
    m_off = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
    b_sim, j_sim = dump(m_sim, True)
    b_off, j_off = dump(m_off, False)

    # 链上要看的 body
    chain_bodies = ["body_link1", "body_link2"] + [f"arm_r_link{i}" for i in range(1, 8)] + \
                   ["arm_r_end_link"]
    chain_joints = [f"idx{61+i}_arm_r_joint{i+1}" for i in range(7)]

    print("=" * 100)
    print("BODY 相对父体的 pos / quat  （pos 单位 m，quat 顺序 wxyz）")
    print("=" * 100)
    print(f"{'body':<18}{'源':<6}{'pos':>42}{'quat':>44}")
    for b in chain_bodies:
        for tag, src in (("仿真", b_sim), ("离线", b_off)):
            if b not in src:
                print(f"{b:<18}{tag:<6}  ** 该模型里没有这个 body **")
                continue
            p, q = src[b]
            print(f"{b:<18}{tag:<6}{str(np.round(p, 5).tolist()):>42}"
                  f"{str(np.round(q, 5).tolist()):>44}")
        if b in b_sim and b in b_off:
            dp = np.linalg.norm(b_sim[b][0] - b_off[b][0])
            dq = min(np.linalg.norm(b_sim[b][1] - b_off[b][1]),
                     np.linalg.norm(b_sim[b][1] + b_off[b][1]))
            mark = "   <<< 不一致" if (dp > 1e-6 or dq > 1e-6) else ""
            print(f"{'':<18}{'差异':<6}{dp:>38.6f}m{dq:>40.6f}{mark}")
        print()

    print("=" * 100)
    print("JOINT  axis / pos / range （range 显示为度）")
    print("=" * 100)
    for jn in chain_joints:
        for tag, src in (("仿真", j_sim), ("离线", j_off)):
            if jn not in src:
                print(f"{jn:<24}{tag:<6}  ** 该模型里没有这个关节 **")
                continue
            a, p, r = src[jn]
            print(f"{jn:<24}{tag:<6}axis={str(np.round(a, 5).tolist()):>26} "
                  f"pos={str(np.round(p, 5).tolist()):>26} "
                  f"range=[{np.degrees(r[0]):9.3f},{np.degrees(r[1]):9.3f}]°")
        if jn in j_sim and jn in j_off:
            da = np.linalg.norm(j_sim[jn][0] - j_off[jn][0])
            dpp = np.linalg.norm(j_sim[jn][1] - j_off[jn][1])
            dr = np.linalg.norm(j_sim[jn][2] - j_off[jn][2])
            mark = "   <<< 不一致" if (da > 1e-6 or dpp > 1e-6 or dr > 1e-6) else ""
            print(f"{'':<24}{'差异':<6}axis={da:>26.6f} pos={dpp:>26.6f} "
                  f"range={np.degrees(dr):>9.3f}°{mark}")
        print()

    # site 相对其父 body 的偏移
    print("=" * 100)
    print("SITE 相对父 body 的 pos / quat")
    print("=" * 100)
    for tag, m, strip in (("仿真", m_sim, True), ("离线", m_off, False)):
        sid = m.site(SITE if strip else "ee_center_site_r").id
        print(f"{tag:<6}pos={str(np.round(m.site_pos[sid], 5).tolist()):>40} "
              f"quat={np.round(m.site_quat[sid], 5).tolist()}")


if __name__ == "__main__":
    main()
