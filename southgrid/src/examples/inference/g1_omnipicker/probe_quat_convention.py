"""state 第 10~14 维的四元数到底是 **xyzw** 还是 **wxyz**？—— 仿真实测，一次定死。

为什么必须查清：此前所有"起始姿态不可达"的结论（131.55° / 125.87° / 41.16°）都是
用 scipy `R.from_quat` 的**默认 xyzw** 算出来的。如果真实约定是 wxyz，那些数字全错，
而且可能把"可达"误判成"不可达"。两处源码给的信号互相矛盾：

  · `orca_gym_local_env.py:851` 的 docstring 写 `"xquat": (4,) [w,x,y,z]`
  · `lerobot_data_storage.py:982` 的 `_G1_OMNIPICKER_STATE_NAMES` 写
    `l_quat_x, l_quat_y, l_quat_z, l_quat_w`（= xyzw），而 `build_state` 对该值**不做重排**

做法：把右臂关节置成一组已知值 → 从**仿真侧独立**算出 ee_center_site_r 相对
g1_omnipicker_body_link1 的旋转矩阵 → 用 scipy 转成两种顺序 → 和 state 里的四个数
逐分量比。谁是零残差谁就是真约定。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_quat_convention.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402


def main() -> None:
    from scipy.spatial.transform import Rotation as R
    from yaml import Loader, load

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
    q_a = np.zeros(7)
    q_b = np.array([2.3709, 0.3702, -0.8897, 1.7229, -0.1114, -1.6065, 1.043])

    for tag, q7 in (("全 0", q_a), ("已知位置解 base_joints", q_b)):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q7)})
        env.mj_forward()

        # (A) 用 storage 的 state 取四元数
        s = np.asarray(storage.build_state(storage.obs_callback(env)),
                       dtype=np.float64).reshape(-1)
        q_state = s[10:14]
        q_obs = np.asarray(storage.obs_callback(env)["/action/end/orientation"],
                           dtype=np.float64).reshape(2, 4)[1]

        # (B) 从仿真侧独立算同一件事：site 相对 base body 的旋转矩阵
        smat = env.query_site_pos_and_mat(["g1_omnipicker_ee_center_site_r"])["g1_omnipicker_ee_center_site_r"]["xmat"]
        Rs = np.asarray(smat, dtype=np.float64).reshape(3, 3)
        _bp, bmat, _bq = env.get_body_xpos_xmat_xquat(["g1_omnipicker_body_link1"])
        Rb = np.asarray(bmat, dtype=np.float64).reshape(3, 3)
        R_rel = Rb.T @ Rs

        q_sci_xyzw = R.from_matrix(R_rel).as_quat()          # scipy 原生：xyzw
        q_wxyz = q_sci_xyzw[[3, 0, 1, 2]]

        print(f"\n===== 右臂关节 = {tag} =====")
        print(f"  state[10:14]           = {np.round(q_state, 6).tolist()}")
        print(f"  obs /end/orientation   = {np.round(q_obs, 6).tolist()}")
        print(f"  两者一致?              {np.allclose(q_state, q_obs, atol=1e-5)}   "
              f"(max|Δ|={np.abs(q_state - q_obs).max():.2e})")
        print(f"  --- 与独立算出的旋转矩阵比 ---")
        e_xyzw = np.abs(q_state - q_sci_xyzw).max()
        e_wxyz = np.abs(q_state - q_wxyz).max()
        # 四元数 ±q 同姿态，两种符号都比
        e_xyzw = min(e_xyzw, np.abs(q_state + q_sci_xyzw).max())
        e_wxyz = min(e_wxyz, np.abs(q_state + q_wxyz).max())
        print(f"  按 xyzw 解释残差 max|Δ| = {e_xyzw:.3e}")
        print(f"  按 wxyz 解释残差 max|Δ| = {e_wxyz:.3e}")
        verdict = "xyzw" if e_xyzw < e_wxyz else "wxyz"
        print(f"  >>> 这一组判定：**{verdict}**（残差小 3 个量级以上才可信）")

    print("\n注：若两组都判同一个约定，即可定死。")


if __name__ == "__main__":
    main()
