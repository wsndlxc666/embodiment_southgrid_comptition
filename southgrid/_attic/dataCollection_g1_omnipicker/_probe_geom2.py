from __future__ import annotations
import os, sys, numpy as np
base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path: sys.path.insert(0, p)
from yaml import Loader, load
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
config_path = os.path.abspath(os.path.join(base_dir, "../common/example.yaml"))
config = load(open(config_path, encoding="utf-8"), Loader=Loader)
_st = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe2")
mgr = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection",
    entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
    default_joint_values={}, obs_callback=_st.obs_callback, env_index=0, device=None,
    scene_manager=SceneManager("localhost:50051", config=config), frame_skip=5,
    orcagym_addr="localhost:50051")
env = mgr.env
def B(name):  # world xpos/xmat of a body by name or id
    try:
        return np.asarray(env.get_body_xpos_xmat_xquat([name])[0]).reshape(3)
    except Exception:
        return None
try:
    m = env.model
    g = m.get_geom_dict()
    print(f"ngeom={m.ngeom} ndict={len(g)}")
    # print schema sample
    for i,(nm,gd) in enumerate(g.items()):
        if i==0: print("schema:", {k:(type(v).__name__) for k,v in gd.items()})
    for nm, gd in g.items():
        sz = np.asarray(gd.get('size', [0]))
        if sz.max() < 0.15: continue
        body_name = gd.get('body') or gd.get('body_name') or gd.get('bodyid')
        bw = B(body_name) if isinstance(body_name, str) else None
        gpos = np.asarray(gd.get('pos', [0,0,0]))
        print(f"  {nm:46s} type={gd.get('type')} size={np.round(sz,4)} body={body_name} body_world={np.round(bw,3) if bw is not None else None} geom_pos={np.round(gpos,4)}")

    base = env.body(agent_conf.base_body)
    print("\n[base]", base, np.round(B(base),4))
    # tool joints current qpos (pos)
    for jn in ["Group_Interactive_Spanner_task_spanner_joint","Group_Interactive_Screwdriver_task_screwdriver_joint","Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint","Group_Interactive_Flashlight_task_flashlight_joint","Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint"]:
        try:
            q = env.get_joint_qpos({jn: None})[jn]
            print("  qpos", jn.split('_task_')[-1], np.round(q[:7],4))
        except Exception as e:
            print("  qpos", jn, "ERR", e)
    # sites
    sd = m.get_site_dict() if hasattr(m,'get_site_dict') else {}
    print("n_site=", len(sd))
    for nm, v in sd.items():
        if any(k in nm for k in ('Tool','Box','EE','ee','task','spanner','screw','knife','flash')):
            print("  site", nm)
finally:
    try: env.close()
    except Exception: pass
