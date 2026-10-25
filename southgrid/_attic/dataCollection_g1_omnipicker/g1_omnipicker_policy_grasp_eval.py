"""单工具任意位姿抓取 · pi0.5-LoRA 策略闭环自测（无落盘）。

用法（先起模型服务，再起本脚本连仿真）：
    # 1) openpi 侧起 websocket serve
    #    cd /home/dgl/lxc/openpi && uv run python scripts/serve_policy.py \
    #        pi05_g1_omnipicker_single_grasp_lora --exp-name=<exp> --checkpoint-step=<N> --port 8000
    # 2) 本脚本
    #    python g1_omnipicker_policy_grasp_eval.py --trials 12 --port 8000 --num_episodes 0

每试：随机 k∈[1,5]、目标工具与可达带内任意 (x,y)、其余在场工具铺开；
从默认中立位开始，由策略闭环驱动 10s；以目标 COM 明显抬离桌面(z>max(0.30,z0+0.12))
判定成功。不写任何 LeRobot 数据。

状态/动作语义与采集一致（18 维，Lerobot G1 OmniPicker）：
  [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
   l_grip_inner_norm, l_grip_outer_norm, r_grip_inner_norm, r_grip_outer_norm]
夹爪列归一化到 [0,1]（按 actuator_ranges 线性），其余列是 B 系绝对末端位姿。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_openpi_client_src = "/home/dgl/lxc/openpi/packages/openpi-client/src"
if _openpi_client_src not in sys.path:
    sys.path.insert(0, _openpi_client_src)

base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    capture_frame_with_idx,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

import g1_omnipicker_collection_scripted_tool_lerobot as _T
import g1_omnipicker_collect_single_grasp_lerobot as C

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_policy_eval_stream"
_L_INIT = [0.0] * 7
_GRASP_OK_Z = 0.30
# 左臂全程中立、左夹爪全开（与采集一致），只让右臂/右爪动。
_G_OPEN = -0.8561
_G_CLOSE = 2.0


def _default_joint_values():
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


class PolicyRolloutDevice:
    """从 websocket 策略拿下一帧动作 → 分解成 OSC 目标 → 写入控制器。

    每 query_every 个 env 步问一次策略；问够 max_queries 或抓到即停。
    update() 由 manager.run_controllers() 每步调用。
    """

    def __init__(self, l_arm, r_arm, l_grip, r_grip, task_status, env, storage,
                 cameras, camera_map, cam_hw, client, prompt,
                 l_gmin, l_gmax, r_gmin, r_gmax,
                 query_every: int = 10, max_queries: int = 40):
        self.l_arm, self.r_arm = l_arm, r_arm
        self.l_grip, self.r_grip = l_grip, r_grip
        self.task_status = task_status
        self.env = env
        self.storage = storage
        self.cameras = cameras
        self.camera_map = camera_map
        self.cam_hw = cam_hw
        self.client = client
        self.prompt = prompt
        self.l_gmin, self.l_gmax = l_gmin, l_gmax
        self.r_gmin, self.r_gmax = r_gmin, r_gmax
        self.query_every = max(1, int(query_every))
        self.max_queries = int(max_queries)
        self._n = 0
        self._queries = 0
        self.done = False
        self.first_latency = None

    def update(self) -> None:
        if self.done:
            return
        self._n += 1
        if self._queries >= self.max_queries:
            self.done = True
            return
        if self._n % self.query_every != 0:
            return
        t0 = time.time()
        try:
            imgs, _ = capture_frame_with_idx(self.cameras, self.camera_map, self.cam_hw)
            obs = self.storage.obs_callback(self.env)
            state = self.storage.build_state(obs).astype(np.float32)
            payload = {
                "images": {
                    "cam_head": np.asarray(imgs["cam_head"]).astype(np.uint8),
                    "cam_wrist_r": np.asarray(imgs["cam_wrist_r"]).astype(np.uint8),
                },
                "state": state,
                "prompt": self.prompt,
            }
            out = self.client.infer(payload)
            a = np.asarray(out["actions"], dtype=np.float32)  # (H, >=18)
            if a.ndim != 2 or a.shape[1] < 18:
                raise ValueError(f"policy actions 形状异常: {a.shape}")
            a0 = a[0, :18]
            if self.first_latency is None:
                self.first_latency = (time.time() - t0) * 1000
            self._queries += 1
        except Exception as e:
            print(f"  [policy] 推理失败: {e}", flush=True)
            self.done = True
            return

        l_pos, l_quat = a0[0:3], a0[3:7]
        r_pos, r_quat = a0[7:10], a0[10:14]
        l_g = a0[14:16] * (self.l_gmax - self.l_gmin) + self.l_gmin
        r_g = a0[16:18] * (self.r_gmax - self.r_gmin) + self.r_gmin
        self.l_arm.update_action_position(l_pos)
        self.l_arm.update_action_axisangle(l_quat)
        self.r_arm.update_action_position(r_pos)
        self.r_arm.update_action_axisangle(r_quat)
        self.l_grip.update_ctrl(np.clip(l_g, self.l_gmin, self.l_gmax))
        self.r_grip.update_ctrl(np.clip(r_g, self.r_gmin, self.r_gmax))

    def force_end(self) -> None:
        self.done = True


def _denorm_range(conf_side):
    n = len(conf_side["actuator_names"])
    rs = conf_side["actuator_ranges"][:n]
    lo = np.array([r[0] for r in rs], dtype=np.float32)
    hi = np.array([r[1] for r in rs], dtype=np.float32)
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 单工具抓取策略闭环自测")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_queries", type=int, default=45, help="策略询问次数上限")
    ap.add_argument("--query_every", type=int, default=10, help="每 N 个 env 步问一次策略")
    ap.add_argument("--task_config", type=str, default="../common/example.yaml")
    ap.add_argument("--placeable_scan", type=str, default="/tmp/placeable_scan.json")
    ap.add_argument("--settle_steps", type=int, default=180)
    ap.add_argument("--fixed_target", type=str, default="",
                    help="可选：固定目标 '工具名:x,y:k' 调试点，如 '扳手:0.55,-0.30:3'")
    args = ap.parse_args()

    from openpi_client import websocket_client_policy as _wcp
    client = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[策略] 已连接 {args.host}:{args.port}", flush=True)

    from yaml import Loader, load as yload
    with open(os.path.join(base_dir, args.task_config), encoding="utf-8") as f:
        scene_config = yload(f, Loader=Loader)
    sm = SceneManager("localhost:50051", config=scene_config)

    dvals = _default_joint_values()
    storage = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_eval_scratch")
    manager = DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=sm, data_storage=None, frame_skip=5, orcagym_addr="localhost:50051",
    )
    env = manager.env
    manager.save_video = False

    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        print("场景初始化失败"); return
    env.set_default_joint_values(dvals)
    manager.set_disable_actuator_group([agent_conf.positions_group])
    base_body = env.body(agent_conf.base_body)

    ctrl_l = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r, init_r)
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6) * 220.0
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)

    l_gn = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gn = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gn, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gn, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_l, agent_conf.base_body,
                                                    l_gn, init_lg, Controller2F85Reverse.ControllerType.DATA)
    r_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_r, agent_conf.base_body,
                                                    r_gn, init_rg, Controller2F85Reverse.ControllerType.DATA)
    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    camera_map = dict(DEFAULT_CAMERA_MAP)
    cameras = {}
    cam_hw = DEFAULT_HW
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        print("camera init fail", e)
    if not cameras:
        print("no cameras; abort"); return

    guard = C._box_guard_xy_minmax(env, base_body)
    cells = C._load_placeable_cells(args.placeable_scan, x_lo=0.42, x_hi=0.80, y_lo=-0.60, y_hi=-0.07)
    if guard is not None:
        cells = [c for c in cells if not C._cell_in_box(c, guard)]
    if len(cells) < 10:
        print(f"可用格点不足 {len(cells)}"); return

    l_gmin, l_gmax = _denorm_range(agent_conf.gripper_l)
    r_gmin, r_gmax = _denorm_range(agent_conf.gripper_r)
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))

    ok = fail = 0
    stat = {}
    t_all = time.time()
    for tr in range(args.trials):
        # 布局：默认随机；也可固定调试点
        if args.fixed_target.strip():
            name, xy_k = args.fixed_target.split(":", 1)
            xys, k = xy_k.split(":")
            x, y = (float(v) for v in xys.split(","))
            target = C.TOOL_NAMES.index(name)
            k = int(k)
            present = [target]
            cand = [t for t in range(5) if t != target]
            rng.shuffle(cand)
            xy = {target: (x, y)}
            occ = [(x, y)]
            for t in cand[: k - 1]:
                c = None
                for _ in range(60):
                    c = C._pick_cell(rng, cells, occ, 0.17)
                    if c is not None and not C._cell_in_box(c, guard):
                        break
                    c = None
                if c is not None:
                    xy[t] = c; occ.append(c)
            present = [t for t in [target] + cand if t in xy]
        else:
            layout = C._sample_episode_layout(rng, cells, guard, list(range(5)))
            if layout is None:
                print("  [skip] layout None"); continue
            target, present, xy = layout

        env.reset(); time.sleep(0.05)
        if not manager.update_scene():
            print("场景更新失败，停止"); break
        env.set_default_joint_values(dvals)
        base_body = env.body(agent_conf.base_body)
        C._place_scene_tools(env, base_body, present, xy)
        C._settle(env, manager, args.settle_steps)
        env.set_default_joint_values(dvals); env.mj_forward()
        base_body = env.body(agent_conf.base_body)
        actual = _T._query_tool_positions_b(env, base_body)
        z0 = float(actual[target, 2])
        if not (0.118 <= z0 <= 0.20):
            print(f"  [skip] 目标静置 z={z0:.3f} 异常，换布局", flush=True)
            continue
        # 干扰物是否掉下桌
        present_z = [float(actual[t, 2]) for t in present]
        if any(z < 0.095 for z in present_z):
            print(f"  [skip] 有工具掉桌下 {present_z}", flush=True)
            continue

        prompt = C._tool_prompt(target, rng)
        layout_txt = " | ".join(f"{C.TOOL_NAMES[t]}@({xy[t][0]:.2f},{xy[t][1]:.2f})" for t in present)
        print(f"\n>>> [trial{tr + 1}] 抓取 {C.TOOL_NAMES[target]} | 在场 {len(present)}: {layout_txt}", flush=True)
        print(f"  prompt: {prompt}  (z0={z0:.3f})", flush=True)

        # 控制器初始化：左爪开、右爪开、左右臂目标=当前末端，让 OSC 稳定起步
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        for c in manager.controllers:
            c.reset()
        ee_b = env.query_site_pos_and_quat_B(
            [agent_conf.l_arm["ee_site_name"], agent_conf.r_arm["ee_site_name"]],
            [agent_conf.base_body],
        )
        # 以当前实际末端位初始化（避免 OSC 起步乱跳）
        def _ee(name):
            return np.asarray(ee_b[name]["xpos"], dtype=np.float32), \
                np.asarray(ee_b[name]["xquat"][[1, 2, 3, 0]], dtype=np.float32)
        lp, lq = _ee(agent_conf.l_arm["ee_site_name"])
        rp, rq = _ee(agent_conf.r_arm["ee_site_name"])
        l_arm.update_action_position(lp)
        l_arm.update_action_axisangle(lq)
        r_arm.update_action_position(rp)
        r_arm.update_action_axisangle(rq)
        l_grip.update_ctrl(np.array([_G_OPEN, _G_OPEN], dtype=np.float32))
        r_grip.update_ctrl(np.array([_G_OPEN, _G_OPEN], dtype=np.float32))
        for _ in range(10):
            action = manager.run_controllers()
            env.step(action)
            env.render()

        dev = PolicyRolloutDevice(
            l_arm, r_arm, l_grip, r_grip, task_status, env, storage,
            cameras, camera_map, cam_hw, client, prompt,
            l_gmin, l_gmax, r_gmin, r_gmax,
            query_every=args.query_every, max_queries=args.max_queries,
        )
        manager.set_device(dev)
        base_body = env.body(agent_conf.base_body)

        step_budget = args.max_queries * args.query_every + 200
        raised_at = None
        for s in range(step_budget):
            action = manager.run_controllers()  # 内部调用 dev.update()（按 query_every 节奏）
            env.step(action)
            env.render()
            if (s + 1) % 50 == 0:
                zc = float(_T._query_tool_positions_b(env, base_body)[target, 2])
                if zc > max(_GRASP_OK_Z, z0 + 0.12):
                    raised_at = s + 1
                    dev.force_end()
                    break
            if dev.done:
                break

        # 收尾保持一小段看是否掉落
        for _ in range(20):
            action = manager.run_controllers()
            env.step(action)
            env.render()

        z1 = float(_T._query_tool_positions_b(env, base_body)[target, 2])
        good = z1 > max(_GRASP_OK_Z, z0 + 0.12)
        ok += 1 if good else 0
        fail += 0 if good else 1
        key = f"{C.TOOL_NAMES[target]}|k{len(present)}"
        stat[key] = stat.get(key, [0, 0])
        stat[key][0] += 1
        stat[key][1] += 1 if good else 0
        q = dev._queries
        print(f"  [{'✓' if good else '✗'}] z {z0:.3f}→{z1:.3f} queries={q}"
              + (f" raised@{raised_at}" if raised_at else ""), flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  自测结束: OK {ok} / FAIL {fail} / {args.trials} 试, {time.time() - t_all:.0f}s", flush=True)
    for k, v in sorted(stat.items()):
        print(f"    {k}: {v[1]}/{v[0]}", flush=True)
    print("=" * 60, flush=True)
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n[停止] 中断", flush=True)
    finally:
        os._exit(0)
