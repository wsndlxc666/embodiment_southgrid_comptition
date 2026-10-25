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
_st = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_api")
mgr = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection",
    entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
    default_joint_values={}, obs_callback=_st.obs_callback, env_index=0, device=None,
    scene_manager=SceneManager("localhost:50051", config=config), frame_skip=5,
    orcagym_addr="localhost:50051")
env = mgr.env
try:
    m, d = env.model, env.data
    print("env.model type:", type(m).__name__)
    print("model has geom-ish:", [a for a in dir(m) if 'geom' in a.lower()][:40])
    print("model has site-ish:", [a for a in dir(m) if 'site' in a.lower()][:40])
    print("env.data type:", type(d).__name__)
    print("data geom-ish:", [a for a in dir(d) if 'geom' in a.lower()][:40])
    print("data site-ish:", [a for a in dir(d) if 'site' in a.lower()][:20])
    print("env query-ish:", [a for a in dir(env) if any(k in a.lower() for k in ['geom','site','body','joint','state'])][:80])
finally:
    try: env.close()
    except Exception: pass
