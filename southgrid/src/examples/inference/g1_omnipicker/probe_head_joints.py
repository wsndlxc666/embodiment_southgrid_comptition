"""头部关节现在是几度？它决定头部相机的朝向，而 state 里没有它。

为什么查这个：评测帧 0 的头相机画面和训练帧 0 **完全不像**（相关 +0.13，训练集
100 集彼此最低 +0.933），而且把右臂摆到训练帧 0 的**同一个位姿**（实测 0.00mm /
0.00°）之后**仍然不像**（[[orca-sim-robot-model-differs-from-xml]] 那条的模型差异
是已知原因之一）。头相机挂在 `head_link2 → head_link1 → body_link2` 上，
中间有两个**只影响头、完全不影响手臂**的关节：

    g1_omnipicker_idx11_head_joint1   hinge  axis=z  range=[-90, +90]   头左右转
    g1_omnipicker_idx12_head_joint2   hinge  axis=z  range=[-20, +30]

22 维 state 里**没有任何一维**记录这两个关节，采集脚本和推理脚本里也都**没有**
显式设置它们 —— 也就是说两边都靠"场景默认值"。如果默认值在两次加载之间会变，
头相机的朝向就会变，而模型收到的 proprio 却一模一样 → 视觉输入与本体感知自相矛盾。

本脚本打印：头两个关节的当前角度、各自限位、以及头相机 body 的世界位姿，
外加 base body 的世界位姿，用来判断"头相对 base 转了多少"。
再顺带把工具/盒子的世界坐标打出来，确认场景几何没变。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_head_joints.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402
from run_task3_tool_storage_scored import (  # noqa: E402
    ROBOT_BASE_BODY, SCORE_PHASES, TOOLBOX_SITE)

HEAD_JOINTS = ["g1_omnipicker_idx11_head_joint1", "g1_omnipicker_idx12_head_joint2"]
HEAD_CAM = "g1_omnipicker_camera_head_body1"
WRIST_CAM = "g1_omnipicker_camera_right_body1"


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

    print("\n=== 头部关节 ===")
    qd = env.query_joint_qpos(HEAD_JOINTS)          # 返回的是 dict，不是数组
    for n in HEAD_JOINTS:
        try:
            v = np.asarray(qd[n], dtype=np.float64).reshape(-1)
            rng = np.asarray(env.gym._mjModel.jnt_range[env.gym._mjModel.joint(n).id],
                             dtype=np.float64)
            print(f"  {n}: {np.degrees(v[0]):+8.2f}°   限位 "
                  f"[{np.degrees(rng[0]):+.1f}, {np.degrees(rng[1]):+.1f}]°")
        except Exception as exc:
            print(f"  {n}: 读不到（{type(exc).__name__}: {exc}）")
    # 全身关节里，凡是**不在** 22 维 state 里的，都是"模型看不见的自由度"
    try:
        allj = [env.gym._mjModel.joint(i).name for i in range(env.gym._mjModel.njnt)]
        print(f"  （仿真共 {len(allj)} 个关节；head/arm 之外的都左右不了 state）")
    except Exception:
        pass

    print("\n=== 相机 / base 的世界位姿 ===")
    for nm in (ROBOT_BASE_BODY, HEAD_CAM, WRIST_CAM):
        try:
            import mujoco
            bid = env.gym._mjModel.body(nm).id
            d = env.gym._mjData
            pos = np.asarray(d.xpos[bid], dtype=np.float64)
            mat = np.asarray(d.xmat[bid], dtype=np.float64).reshape(3, 3)
            from scipy.spatial.transform import Rotation as R
            eul = np.degrees(R.from_matrix(mat).as_euler("xyz"))
            # 相机轴向：mujoco 相机沿自身 -z 看，这里只看 body 的 x/y/z 朝向
            print(f"  {nm}\n      pos = {np.round(pos, 4).tolist()}"
                  f"\n      euler(xyz,deg) = {np.round(eul, 2).tolist()}")
        except Exception as exc:
            print(f"  {nm}: 读不到（{type(exc).__name__}: {exc}）")

    print("\n=== base → 相机 的相对朝向（这才是与采集可比的东西）===")
    try:
        import mujoco  # noqa: F401
        d = env.gym._mjData
        m = env.gym._mjModel
        b = m.body(ROBOT_BASE_BODY).id
        for nm in (HEAD_CAM, WRIST_CAM):
            c = m.body(nm).id
            Rb = np.asarray(d.xmat[b], dtype=np.float64).reshape(3, 3)
            Rc = np.asarray(d.xmat[c], dtype=np.float64).reshape(3, 3)
            from scipy.spatial.transform import Rotation as R
            eul = np.degrees(R.from_matrix(Rb.T @ Rc).as_euler("xyz"))
            rel = Rb.T @ (np.asarray(d.xpos[c]) - np.asarray(d.xpos[b]))
            print(f"  {nm}: base 系位移 {np.round(rel, 4).tolist()}"
                  f"  相对 euler(xyz,deg) {np.round(eul, 2).tolist()}")
    except Exception as exc:
        print(f"  相对量算不出（{type(exc).__name__}: {exc}）")

    print("\n=== 工具/盒子世界坐标（确认场景几何没变）===")
    sites = [p["site"] for p in SCORE_PHASES] + [TOOLBOX_SITE]
    try:
        pos, _quat = env.query_site_pos_and_quat(sites)
        for s, p in zip(sites, pos):
            print(f"  {s:60s} {np.round(p, 4).tolist()}")
    except Exception as exc:
        print(f"  读不到（{type(exc).__name__}: {exc}）")


if __name__ == "__main__":
    main()
