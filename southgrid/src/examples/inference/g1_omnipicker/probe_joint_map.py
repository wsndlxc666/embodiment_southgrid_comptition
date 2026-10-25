"""离线 MJCF 的 FK 和**仿真里的机器人**是不是同一个映射？—— ik 模式失败的根因排查。

2026-09-17 的症状：`--start_align ik` 把右臂 teleport 到离线 IK 解，三次迭代读数
完全一致（可复现，不是抖动），但残差是 **510.41mm / 101.32°**，而离线 FK 预测同一组
关节应当给出 13.64mm / 0.16°。差了 510mm。

这只有两种可能：
  (a) `env.set_joint_qpos` 没真正写进去（或写进去又被覆盖）；
  (b) **离线 `model/G1_omnipicker/g1_omnipicker.xml` 和仿真加载的模型不是同一个**
      —— 仿真加载的是 orcagym 生成的场景 XML（/home/dgl/.orcagym/tmp/*.xml），
      机器人挂在场景里，根体/基座姿态可能与独立加载机器人 XML 时不同。

本脚本一次判死：在仿真里设一组关节 → 读**实时**末端位姿 → 和离线 FK 对同一组关节
的预测逐分量比。同时把 qpos 读回来确认 (a)。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_joint_map.py
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

R_JOINTS = [f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)]
XML = os.path.join(base_dir, "..", "..", "..", "model", "G1_omnipicker", "g1_omnipicker.xml")
OFFLINE_BASE_BODY = "body_link1"
OFFLINE_EE_SITE = "ee_center_site_r"

# 三组测试关节（弧度）
CASES = {
    "全 0": np.zeros(7),
    "已知位置解 base_joints": np.array([2.3709, 0.3702, -0.8897, 1.7229, -0.1114, -1.6065, 1.043]),
    "ik basin#5": np.array([0.56471, 2.04204, -0.23143, 1.71851, -0.32236, -1.74, 0.68399]),
}


def offline_fk() -> dict:
    """离线 MJCF 的 FK：site 相对基座 body 的位置 + 姿态（四元数 xyzw）。"""
    import mujoco
    from scipy.spatial.transform import Rotation as R
    m = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in R_JOINTS]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, OFFLINE_EE_SITE)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OFFLINE_BASE_BODY)
    out = {}
    for tag, q7 in CASES.items():
        d.qpos[:] = 0.0
        d.qpos[qadr] = q7
        mujoco.mj_forward(m, d)
        p = d.site_xpos[sid] - d.xpos[bid]
        Rb = d.xmat[bid].reshape(3, 3)
        Rs = d.site_xmat[sid].reshape(3, 3)
        out[tag] = (p, R.from_matrix(Rb.T @ Rs).as_quat())
    return out


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from yaml import Loader, load

    off = offline_fk()
    print("离线 MJCF FK 预测（site 相对 base body）:")
    for tag, (p, q) in off.items():
        print(f"  {tag:22s} pos={np.round(p, 4).tolist()}  quat={np.round(q, 4).tolist()}")

    cfg_path = os.path.abspath(os.path.join(base_dir, "../../dataCollection/common/example.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = E.SceneManager("localhost:50051", config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr="localhost:50051")
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset()
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    # 官方口径：query_site_pos_and_quat_B（probe_base_body 用它对到过 2e-4）
    def official():
        q = env.query_site_pos_and_quat_B([EE_SITE_R], [ROBOT_BASE_BODY])[EE_SITE_R]
        p = np.asarray(q["xpos"], dtype=np.float64).reshape(3)
        xq = np.asarray(q["xquat"], dtype=np.float64).reshape(4)[[1, 2, 3, 0]]  # wxyz->xyzw
        return p, R.from_quat(xq)

    # 手算口径：就是 runner 里 _live_ee_b 用的那条路
    def manual():
        smat = env.query_site_pos_and_mat([EE_SITE_R])[EE_SITE_R]["xmat"]
        Rs = np.asarray(smat, dtype=np.float64).reshape(3, 3)
        spos = np.asarray(env.query_site_pos_and_quat([EE_SITE_R])[EE_SITE_R]["xpos"],
                          dtype=np.float64).reshape(3)
        bpos, bmat, bq = env.get_body_xpos_xmat_xquat([ROBOT_BASE_BODY])
        Rb = np.asarray(bmat, dtype=np.float64).reshape(3, 3)
        bp = np.asarray(bpos, dtype=np.float64).reshape(3)
        return (Rb.T @ (spos - bp)), R.from_matrix(Rb.T @ Rs), (bp, Rs, spos,
                                                                np.asarray(bq, dtype=np.float64))

    print(f"\n仿真侧（base body = {ROBOT_BASE_BODY}, site = {EE_SITE_R}）:")
    for tag, q7 in CASES.items():
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q7)})
        env.mj_forward()

        p_off, q_off = off[tag]
        p_o, R_o = official()
        p_m, R_m, extra = manual()
        bp, Rs, spos, bq = extra

        # qpos 读回：local env 有 query_joint_qpos（dict 入参）
        got = None
        for fn, mk in (("query_joint_qpos", lambda n: [n]),
                       ("query_joint_qpos", lambda n: {n: None}.keys()),
                       ("get_joint_qpos", lambda n: [n])):
            f = getattr(env, fn, None)
            if f is None:
                continue
            try:
                d = f(list(E.agent_conf.r_arm["joint_names"]))
                if isinstance(d, dict):
                    got = np.array([float(np.asarray(d[k]).reshape(-1)[0])
                                    for k in E.agent_conf.r_arm["joint_names"]])
                else:
                    got = np.asarray(d, dtype=np.float64).reshape(-1)[:7]
                break
            except Exception as e:
                last = repr(e)[:60]
                continue

        print(f"\n--- {tag} ---")
        print(f"  离线 FK            pos={np.round(p_off, 4).tolist()}  quat={np.round(q_off, 4).tolist()}")
        print(f"  仿真 official(_B)  pos={np.round(p_o, 4).tolist()}   Δpos={np.linalg.norm(p_o-p_off)*1000:7.2f}mm")
        print(f"  仿真 manual(手算)  pos={np.round(p_m, 4).tolist()}   Δpos={np.linalg.norm(p_m-p_off)*1000:7.2f}mm")
        print(f"  official vs manual 差 = {np.linalg.norm(p_o-p_m)*1000:.2f} mm"
              f"   ← 非 0 就说明 _live_ee_b 那条路本身写错了")
        print(f"  现场量: base 世界 pos={np.round(bp, 4).tolist()}  base quat(wxyz)={np.round(bq, 4).tolist()}")
        print(f"          site 世界 pos={np.round(spos, 4).tolist()}")
        print(f"          base 旋转矩阵 R=\n{np.round(Rb_dbg(Rs), 4)}")
        chk = "读不到" if got is None else \
              ("是" if np.allclose(got, q7, atol=1e-5) else f"**否** 读回 {np.round(got, 4).tolist()}")
        print(f"  qpos 读回 = 设定值? {chk}")

    print("\n判读：")
    print("  · official 与离线 FK 一致 → 离线 IK 有效；若 manual 与之不符，就是 _live_ee_b 写错了。")
    print("  · official 也与离线 FK 差几百 mm → 模型确实不是同一个，离线 IK 结论作废。")


def Rb_dbg(Rs):
    return Rs


if __name__ == "__main__":
    main()
