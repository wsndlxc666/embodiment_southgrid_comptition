"""可达性探针：对桌面网格逐格做「扳手单抓抬升」，无相机无落盘。

输出每格 (x,y) 扳手静置高度 z0 与抓后高度 z1 及 OK/FAIL。
判断规则与采集一致：z1 > max(0.30, z0+0.12) 视为可抓。
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path: sys.path.insert(0, p)
from yaml import Loader, load
from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller, create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

import g1_omnipicker_collection_scripted_tool_lerobot as _T
import g1_omnipicker_collect_single_grasp_lerobot as C
import data_collection_scripted as scripted

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
_L_INIT = [0.0] * 7
# 直接解析扫描 JSON，取全表静置良好格点，探针含 x 到 0.90 找可达前沿
_grid = json.load(open('/tmp/placeable_scan.json', encoding='utf-8'))
_cells0 = []
for _k, _v in _grid.items():
    if _v and float(_v['rest'][2]) >= 0.125 and float(_v['dxy']) < 0.05:
        _x, _y = (float(s) for s in _k.split(','))
        if 0.40 <= _x <= 0.90 and -0.62 <= _y <= -0.05:
            _cells0.append((round(_x, 3), round(_y, 3)))
CELLS = sorted(set(_cells0))
print(f"[probe] 全表候选 {len(CELLS)} 格", flush=True)

def main():
    cfg = load(open(os.path.join(base_dir, "../common/example.yaml"), encoding="utf-8"), Loader=Loader)
    sm = SceneManager("localhost:50051", config=cfg)
    _dummy_storage = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_reach_probe_scratch")
    mgr = DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=ENTRY_POINT,
        default_joint_values={}, obs_callback=_dummy_storage.obs_callback, env_index=0, device=None,
        scene_manager=sm, data_storage=None, frame_skip=5, orcagym_addr="localhost:50051",
    )
    env = mgr.env
    mgr.save_video = False

    env_dt = float(mgr.time_step) * int(mgr.frame_skip)
    steps_scale = max(1, int(round((1.0 / 20.0) / env_dt)))
    def _cs(frames): return max(1, int(round(float(frames) * steps_scale)))

    default_joint_values = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    env.reset(); time.sleep(0.1)
    mgr.update_scene(); env.set_default_joint_values(default_joint_values)
    mgr.set_disable_actuator_group([agent_conf.positions_group])
    base = env.body(agent_conf.base_body)

    ctrl_l = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r, init_r)
    for a in (l_arm, r_arm):
        a.controller.kp = np.ones(6) * 220.0
        a.controller.kd = 2.0 * np.sqrt(a.controller.kp)
    l_gn = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gn = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gn, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gn, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_l, agent_conf.base_body, l_gn, init_lg, Controller2F85Reverse.ControllerType.DATA)
    r_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_r, agent_conf.base_body, r_gn, init_rg, Controller2F85Reverse.ControllerType.DATA)
    for c_ in (l_arm, r_arm, l_grip, r_grip):
        mgr.add_controller(c_)
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    mgr.set_task_status_controller(task_status)
    mgr.set_task(EmptyTask(env))

    guard = C._box_guard_xy_minmax(env, base)
    td = C.tool_data if hasattr(C, "tool_data") else None

    wps = _T._load_waypoint_yaml(os.path.join(base_dir, "my_waypoint_tool1.yaml"))
    g_open, g_close = wps["g_open"], wps["g_close"]
    wrench_ref = C.TOOL_REF_POS_B[0]

    results = {}
    t0 = time.time()
    ok = fail = skip = 0
    for (x, y) in CELLS:
        cell = (round(x, 3), round(y, 3))
        if guard is not None and C._cell_in_box((x, y), guard):
            results[f"{x},{y}"] = {"z0": None, "z1": None, "res": "BOX"}
            skip += 1
            continue
        env.reset(); time.sleep(0.05)
        if not mgr.update_scene():
            print("scene update failed; abort"); break
        env.set_default_joint_values(default_joint_values)
        base = env.body(agent_conf.base_body)
        C._place_scene_tools(env, base, [0], {0: (x, y)})
        C._settle(env, mgr, 200)
        env.set_default_joint_values(default_joint_values); env.mj_forward()
        actual = _T._query_tool_positions_b(env, base)
        z0 = float(actual[0, 2])
        if not (0.118 <= z0 <= 0.20):
            results[f"{x},{y}"] = {"z0": round(z0, 4), "z1": None, "res": "LOW"}
            print(f"P {x:.2f},{y:.2f} z0={z0:.3f} LOW", flush=True)
            fail += 1
            continue
        delta = actual - C.TOOL_REF_POS_B
        warped = _T._warp_waypoints(wps, delta[0])
        segs = C._grasp_lift_segments(warped["waypoints"], safe_z=0.48,
            f={"transit":45,"descend":25,"grasp":30,"settle":28,"lift":30,"hold":25}, ctrl_steps=_cs)
        try:
            l_pos, l_quat, r_pos, r_q, l_gm, r_gm = scripted.build_segmented_trajectory(
                env, agent_conf, list(segs), g_open, g_close)
        except Exception as e:
            results[f"{x},{y}"] = {"z0": round(z0, 4), "z1": None, "res": f"ERR:{e}"}
            print(f"P {x:.2f},{y:.2f} build ERR {e}", flush=True)
            fail += 1
            continue
        dev = _T.G1ScriptedTrajectoryDevice(l_arm, r_arm, l_grip, r_grip, task_status,
            l_pos, l_quat, r_pos, r_q, l_gm, r_gm, place_arm_at=None, box_bottom=None)
        mgr.set_device(dev)
        try:
            mgr.run_episode()
        except Exception as e:
            results[f"{x},{y}"] = {"z0": round(z0, 4), "z1": None, "res": f"RUNERR:{e}"}
            print(f"P {x:.2f},{y:.2f} run ERR {e}", flush=True)
            fail += 1
            continue
        z1 = float(_T._query_tool_positions_b(env, base)[0, 2])
        good = z1 > max(0.30, z0 + 0.12)
        results[f"{x},{y}"] = {"z0": round(z0, 4), "z1": round(z1, 4), "res": "OK" if good else "FAIL"}
        print(f"P {x:.2f},{y:.2f} z0={z0:.3f} z1={z1:.3f} {'OK' if good else 'FAIL'}", flush=True)
        if good: ok += 1
        else: fail += 1
    print(f"[done] OK {ok} / FAIL {fail} / BOX {skip} / {len(CELLS)} cells in {time.time()-t0:.0f}s", flush=True)
    json.dump(results, open("/tmp/grasp_reach_map.json", "w"), indent=1)
    try:
        env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
    os._exit(0)
