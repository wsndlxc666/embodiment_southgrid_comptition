"""藏匿工具可见性 A/B 测试（无落盘，仅渲染比较）。

场景固定：扳手在场 @(0.55,-0.30)，其余 4 件为"不在场工具"。
S0：不在场工具按当前方案丢到 x=0.35 桌边掉落列（物理沉降后渲染）。
B ：不在场工具瞬移到地面以下深处（地板遮挡，必不可见）作为基准，静态渲染。

若 S0 与 B 的头/腕相机帧几乎一致，说明当前藏法不可见，可放心用于采集。
输出每个候选藏点与基准的逐像素差异均值与显著差异像素占比。
"""
from __future__ import annotations
import os, sys
base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path: sys.path.insert(0, p)
import time
import numpy as np
from yaml import Loader, load
from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_task import TaskStatusController
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (DEFAULT_CAMERA_MAP, DEFAULT_HW, bring_up_cameras,
                                        close_cameras, probe_camera_hw, capture_frame_with_idx)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask
import g1_omnipicker_collect_single_grasp_lerobot as C
import g1_omnipicker_collection_scripted_tool_lerobot as _T

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_stash_test_stream"
_L_INIT = [0.0] * 7
PRESENT = [0]
XY = {0: (0.55, -0.30)}
STASH_CELLS = [(C._STASH_X, y) for y in C._STASH_YS]  # x=0.35 掉落列
GROUND_Z_BASE = -0.75  # 地面以下足够深，保证被地板遮挡


def _default_joint_values():
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


def _render(cameras, camera_map, hw):
    imgs, _ = capture_frame_with_idx(cameras, camera_map, hw)
    return {k: np.asarray(v).astype(np.float32) for k, v in imgs.items()}


def _place_absent_under_floor(env, base_body):
    """把不在场工具放到地面以下静态位（仅供单帧渲染基准）。"""
    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos).reshape(3)
    qw = np.asarray(base_quat_wxyz).reshape(4)
    Rm = _T._rotmat_from_wxyz(qw) if hasattr(_T, "_rotmat_from_wxyz") else None
    from scipy.spatial.transform import Rotation as _R
    base_rot = _R.from_quat(qw[[1, 2, 3, 0]])
    qpos = {}
    for t in range(5):
        if t in PRESENT:
            continue
        pos_b = np.array([0.6, -0.3 - 0.15 * t, GROUND_Z_BASE])
        world = base_pos + base_rot.apply(pos_b)
        rot = base_rot * _R.from_quat(np.asarray(C.TOOL_REF_QUAT_XYZW_B[t]))
        wq = rot.as_quat()
        qpos[C.TOOL_JOINTS[t]] = np.concatenate([world, wq[[3, 0, 1, 2]]])
    env.set_joint_qpos(qpos)
    env.mj_forward()


def diff_stats(a: dict, b: dict):
    out = {}
    for k in a:
        da = np.abs(a[k] - b[k]).mean()
        changed = (np.abs(a[k] - b[k]) > 12.0).mean()
        out[k] = (round(float(da), 2), round(float(changed) * 100, 2))
    return out


def main():
    cfg = load(open(os.path.join(base_dir, "../common/example.yaml"), encoding="utf-8"), Loader=Loader)
    sm = SceneManager("localhost:50051", config=cfg)
    _st = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_stash_test_scratch")
    mgr = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection",
        entry_point=ENTRY_POINT, default_joint_values={}, obs_callback=_st.obs_callback,
        env_index=0, device=None, scene_manager=sm, data_storage=None,
        frame_skip=5, orcagym_addr="localhost:50051")
    env = mgr.env
    mgr.save_video = False
    dvals = _default_joint_values()

    env.reset(); time.sleep(0.1); mgr.update_scene(); env.set_default_joint_values(dvals)
    mgr.set_disable_actuator_group([agent_conf.positions_group])
    ts = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    mgr.set_task_status_controller(ts)
    mgr.set_task(EmptyTask(env))

    camera_map = dict(DEFAULT_CAMERA_MAP)
    cameras = {}
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        print("camera init fail", e)
    if not cameras:
        print("no cameras; abort"); return

    # 单场景 A/B：一次沉降，先渲染 S0(不在场工具丢 x0.35 列)，随后仅把不在场工具
    # 瞬移到地下深处(同 episode、同背景)，再渲染。两次帧差只来自藏点鬼影。
    env.reset(); time.sleep(0.05); mgr.update_scene(); env.set_default_joint_values(dvals)
    base = env.body(agent_conf.base_body)
    C._place_scene_tools(env, base, PRESENT, XY)
    C._settle(env, mgr, 220)
    env.set_default_joint_values(dvals); env.mj_forward()
    base = env.body(agent_conf.base_body)
    actual = _T._query_tool_positions_b(env, base)
    print("  S0 实际落点 pos:", {C.TOOL_NAMES[t]: tuple(round(float(v), 3) for v in actual[t]) for t in range(5)}, flush=True)

    A = _render(cameras, camera_map, hw)
    _place_absent_under_floor(env, base)
    env.mj_forward()
    B = _render(cameras, camera_map, hw)

    stats = diff_stats(A, B)
    print("diff A(S0丢落) vs B(地下不可见):", stats, flush=True)
    for k in A:
        d = np.abs(A[k] - B[k])
        chg = d > 12.0
        if chg.any():
            ys, xs = np.where(chg[..., 0] | chg[..., 1] | chg[..., 2])
            print(f"  {k}: 变化像素 bbox y[{ys.min()}..{ys.max()}] x[{xs.min()}..{xs.max()}] HxW={A[k].shape[:2]}", flush=True)

    # 直接把两帧与差异热图存下备查
    try:
        from PIL import Image
        for k in A:
            Image.fromarray(A[k].astype(np.uint8)).save(f"/tmp/stashA_{k}.png")
            Image.fromarray(B[k].astype(np.uint8)).save(f"/tmp/stashB_{k}.png")
            d = np.abs(A[k].astype(np.int16) - B[k].astype(np.int16)).max(axis=2)
            heat = (d * (255.0 / max(1.0, float(d.max())))).astype(np.uint8)
            Image.fromarray(heat).save(f"/tmp/stashdiff_{k}.png")
    except Exception as e:
        print("save fail", e, flush=True)

    try:
        env.stop_save_video()
    except Exception:
        pass
    close_cameras(cameras)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
    os._exit(0)
