from __future__ import annotations
import os, sys, time, json, numpy as np
base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path: sys.path.insert(0, p)
from yaml import Loader, load
from conf import g1_omnipicker_conf as agent_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
config_path = os.path.abspath(os.path.join(base_dir, "../common/example.yaml"))
config = load(open(config_path, encoding="utf-8"), Loader=Loader)
_st = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_scan_pl")
mgr = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection",
    entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
    default_joint_values={}, obs_callback=_st.obs_callback, env_index=0, device=None,
    scene_manager=SceneManager("localhost:50051", config=config), frame_skip=5,
    orcagym_addr="localhost:50051")
env = mgr.env
base = env.body(agent_conf.base_body)
bp = np.asarray(env.get_body_xpos_xmat_xquat([base])[0]).reshape(3)

# wrench 参考
REF = np.array([0.5654831, -0.0693993, 0.1514528])
JOINT = "Group_Interactive_Spanner_task_spanner_joint"
QUERY=JOINT

def set_default(env):
    env.set_default_joint_values({})
env.set_default_joint_values({})
env.mj_forward()

# figure tool body name from joint if exists
try:
    jt = env.joint(JOINT)
    print("joint ok:", JOINT)
except Exception as e:
    print("joint err", e)

def settle_rest_xyz(x, y, n_steps=80):
    """teleport wrench to base-frame (x,y,z=REF z) then step; return rest pos(base frame)."""
    xp, xm, _ = env.get_body_xpos_xmat_xquat([base])
    xp = np.asarray(xp).reshape(3); xm = np.asarray(xm).reshape(3,3)
    rel = np.array([x, y, REF[2]], dtype=np.float64)
    world = xp + xm @ rel
    q = np.concatenate([world, np.array([0.,0.7071068,0.7071067,0.0])])
    env.set_joint_qpos({JOINT: q})
    env.mj_forward()
    mgr.set_init_ctrl()
    env.set_ctrl(mgr.ctrl)
    for _ in range(n_steps):
        env.step(mgr.ctrl)
    p = np.asarray(env.query_position_body_B(QUERY, base)).reshape(3)
    return p


grid = {}
xs = np.round(np.arange(0.35, 0.95+1e-9, 0.05),3)
ys = np.round(np.arange(-0.62, -0.02+1e-9, 0.05),3)
print("scanning", len(xs)*len(ys), "cells ...")
t0=time.time()
for x in xs:
    for y in ys:
        try:
            p = settle_rest_xyz(x,y)
        except Exception as e:
            grid[f"{x},{y}"] = None; continue
        dz = abs(p[2]-REF[2]); dxy = np.hypot(p[0]-x, p[1]-y)
        grid[f"{x},{y}"] = {"rest": np.round(p,3).tolist(), "dz": round(float(dz),3), "dxy": round(float(dxy),3)}
print(f"scan done in {time.time()-t0:.0f}s")
json.dump(grid, open("/tmp/placeable_scan.json","w"), indent=1)
# 汇总好格子
good = {k:v for k,v in grid.items() if v and v["dz"]<0.03 and v["dxy"]<0.05}
print("GOOD cells:", len(good))
for k in sorted(good): print("  ", k, good[k]["rest"])
env.close()
