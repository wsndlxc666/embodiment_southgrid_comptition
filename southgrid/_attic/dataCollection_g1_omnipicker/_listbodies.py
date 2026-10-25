import os, sys, numpy as np
base_dir = os.path.dirname(os.path.realpath(__file__)); project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path: sys.path.insert(0, p)
from yaml import Loader, load
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
config = load(open(os.path.join(base_dir, "../common/example.yaml"), encoding="utf-8"), Loader=Loader)
_st = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_lb")
mgr = DataCollectionManager(agent_name="g1_omnipicker", env_name="DataCollection", entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv", default_joint_values={}, obs_callback=_st.obs_callback, env_index=0, device=None, scene_manager=SceneManager("localhost:50051", config=config), frame_skip=5, orcagym_addr="localhost:50051")
env = mgr.env
base = env.body(agent_conf.base_body)
try:
    # dump all bodies containing keyword
    nmap = getattr(env.model, "_body_id2name_map", None)
    print("has body map:", nmap is not None)
    import re
    for kw in ["Spanner","Screwdriver","Knife","Flashlight","Electricians"]:
        pass
    # via dir env.model
    attrs=[a for a in dir(env.model) if "body" in a.lower()]
    print("model body attrs:", attrs[:30])
finally:
    env.close()
