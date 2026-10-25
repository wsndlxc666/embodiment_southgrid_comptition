"""只读几何探针：dump 大平面 geom / 工具 / 箱 / 相机，用于确定安全撒放区。
不改动任何状态。"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from yaml import Loader, load
from conf import g1_omnipicker_conf as agent_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

config_path = os.path.abspath(os.path.join(base_dir, "../common/example.yaml"))
with open(config_path, "r", encoding="utf-8") as f:
    config = load(f, Loader=Loader)

sm = SceneManager("localhost:50051", config=config)
_storage = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_geom_scratch")
mgr = DataCollectionManager(
    agent_name="g1_omnipicker",
    env_name="DataCollection",
    entry_point=ENTRY_POINT,
    default_joint_values={},
    obs_callback=_storage.obs_callback,
    env_index=0,
    device=None,
    scene_manager=sm,
    frame_skip=5,
    orcagym_addr="localhost:50051",
)
env = mgr.env
try:
    m = env.model
    d = env.data

    # ---- 机器人 base ----
    base = env.body(agent_conf.base_body)
    bx = np.asarray(env.get_body_xpos_xmat_xquat([base])[0]).reshape(3)
    print(f"\n[base] {base} world pos = {bx.round(4)}")
    print(f"[base] yaw-ish: base→world rotation used by scripts")

    # ---- 找大平面(桌面/货架面) geom ----
    print("\n[geom] 尺寸>0.2 的 box/capsule/cyl（可能的桌面/箱体）:")
    G = m.ngeom
    for i in range(G):
        name = m.geom_name(i)
        typ = m.geom_type[i]
        sz = m.geom_size[i]
        if np.max(sz) < 0.2:
            continue
        wpos = d.geom_xpos[i]
        print(f"  {i:3d} {name} type={typ} size={np.round(sz,4)} world_pos={np.round(wpos,4)}")

    # ---- 工具 joint / 参考 ----
    print("\n[tools] 各工具 free joint 当前 qpos 前7:")
    jnames = [
        "Group_Interactive_Spanner_task_spanner_joint",
        "Group_Interactive_Screwdriver_task_screwdriver_joint",
        "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
        "Group_Interactive_Flashlight_task_flashlight_joint",
        "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
    ]
    for jn in jnames:
        try:
            q = env.get_joint_qpos({jn: None})[jn]
            print(f"  {jn.split('_task_')[-1]:28s} qpos(pos3+quat4)={np.round(q[:7],4)}")
        except Exception as e:
            print(f"  {jn}: ERR {e}")

    # ---- 相机 ----
    print("\n[cam] 场景相机:")
    for i in range(m.ncam):
        print(f"  cam{i} {m.cam_name(i)} pos={np.round(d.cam_xpos[i],3)} fovy={m.cam_fovy[i]:.1f}")

    # ---- 所有 site ----
    print("\n[site] 全部:")
    for i in range(m.nsite):
        print(f"  {m.site_name(i):44s} world={np.round(d.site_xpos[i],4)}")

    # ---- 一帧稳定 ----
    mgr.set_init_ctrl()
    env.set_ctrl(mgr.ctrl)
    env.mj_forward()
    print("\n[done]")
finally:
    try:
        env.close()
    except Exception:
        pass
