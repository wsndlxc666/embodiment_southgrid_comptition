"""G1 OmniPicker 任意位置·单工具抓取脚本化采集。

每集：
  1. 从桌面可放网格随机抽 k∈[1,5] 个互不重叠的位置（k 含目标工具）；
  2. 目标工具名字写入语言指令（如 "抓取扳手"），本集只抓它；
  3. 其余在场工具作为干扰物静止在桌面，不在场工具藏到桌下掉落区；
  4. 脚本化：高位过渡 → 垂直下降到接近点 → 移到抓取点 → 闭爪驻留 → 抬升 → 保持；
  5. 结束后校验目标 z 是否明显抬离桌面，成功才保存，失败 clear_data 重采。

复用 g1_omnipicker_collection_scripted_tool_lerobot 的控制器/存储/相机/设备接线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

OPENPI_DATA_ROOT = os.path.normpath(
    os.path.join(base_dir, "..", "..", "..", "..", "..", "openpi", "data")
)

# 复用工具整理采集脚本里的设备/路点/几何函数与常量。
import g1_omnipicker_collection_scripted_tool_lerobot as _T

import data_collection_scripted as scripted
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
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_single_grasp_lerobot_stream"

orca_logger = get_orca_logger(
    name="G1SingleGraspScripted",
    log_file="g1_omnipicker_collect_single_grasp_lerobot.log",
    max_bytes=20 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)

_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TOOL_NAMES = _T._TOOL_NAMES
TOOL_JOINTS = _T._TOOL_BODY_JOINT_NAMES
TOOL_REF_POS_B = _T._TOOL_REFERENCE_POS_B
TOOL_REF_QUAT_XYZW_B = _T._TOOL_REFERENCE_QUAT_XYZW_B

KNIFE_L_IDX = 2
KNIFE_R_IDX = 4

# 抓取成功判定：目标工具 COM(base 系)抬升超过桌面高度很多即认为已被抓起。
_GRASP_OK_Z = 0.30

# 在桌面上掉落/藏匿用的列（扫描验证：x=0.35 整列都会掉到桌下地面）
_STASH_X = 0.35
_STASH_YS = [-0.60, -0.45, -0.30, -0.15]


# 冒烟用固定布局：(target, {tool: (x,y)})；仅验证链路/几何用，进入正式随机布局前按序消费。
FIXED_SMOKE_LAYOUTS = [
    # k=1：扳手在开放桌面中部
    (0, {0: (0.55, -0.45)}),
    # k=2：目标螺丝刀近参考位，左电工刀作干扰
    (1, {1: (0.58, -0.16), 2: (0.70, -0.50)}),
    # k=5：全在场铺开
    (3, {3: (0.52, -0.50), 0: (0.62, -0.10), 1: (0.78, -0.35), 2: (0.45, -0.35), 4: (0.70, -0.55)}),
    # k=1：左电工刀在非参考位
    (2, {2: (0.68, -0.30)}),
]


def _tool_prompt(tool_idx: int, rng: np.random.Generator) -> str:
    """语言指令模板；名称是辨识关键，句法略作变化增强鲁棒性。"""
    name = TOOL_NAMES[tool_idx]
    r = float(rng.random())
    if r < 0.6:
        return f"抓取{name}"
    if r < 0.8:
        return f"抓起{name}"
    return f"拿起{name}"


def _load_placeable_cells(scan_json: str, x_lo=0.42, x_hi=0.80, y_lo=-0.60, y_hi=-0.07) -> list[tuple[float, float]]:
    """从可放置扫描结果取「稳定在桌面上」的网格中心。

    good = 静置后 z 仍在桌面高度(≥0.12) 且基本没漂移(dxy<0.05)。
    """
    cells: list[tuple[float, float]] = []
    if not os.path.isfile(scan_json):
        return cells
    try:
        with open(scan_json, "r", encoding="utf-8") as f:
            grid = json.load(f)
    except Exception as e:
        orca_logger.warning(f"加载可放置扫描失败 {scan_json}: {e}")
        return cells
    for key, v in grid.items():
        if not v:
            continue
        try:
            x, y = (float(s) for s in key.split(","))
        except Exception:
            continue
        rz = float(v["rest"][2])
        dxy = float(v["dxy"])
        if rz >= 0.12 and dxy < 0.05 and x_lo <= x <= x_hi and y_lo <= y <= y_hi:
            cells.append((x, y))
    # 去重排序
    cells = sorted(set((round(x, 3), round(y, 3)) for x, y in cells))
    orca_logger.info(f"[布局] 可用放置网格: {len(cells)} 个（{scan_json}）")
    return cells


def _box_guard_xy_minmax(env, base_body: str, margin: float = 0.06):
    """返回排除工具箱所需的最小 xy 范围（含 margin）；不可用时返回 None。"""
    try:
        xy_min, xy_max, _ = _T._calibrate_toolbox_xy_aabb(env, base_body)
        return (np.asarray(xy_min) - margin, np.asarray(xy_max) + margin)
    except Exception as e:
        orca_logger.warning(f"[布局] 工具箱范围标定失败，跳过: {e}")
        return None


def _cell_in_box(cell, guard) -> bool:
    if guard is None:
        return False
    (lo, hi) = guard
    return bool(lo[0] <= cell[0] <= hi[0] and lo[1] <= cell[1] <= hi[1])


def _pick_cell(rng, pool, occupied, min_sep):
    """从 pool 中采一个与 occupied 各点距离≥min_sep 的格点（中心可微抖动）。"""
    tries = 120
    for _ in range(tries):
        if not pool:
            return None
        x, y = pool[int(rng.integers(0, len(pool)))]
        if occupied and min( (x - ox) ** 2 + (y - oy) ** 2 for ox, oy in occupied ) < min_sep * min_sep:
            continue
        jx = x + float(rng.uniform(-0.008, 0.008))
        jy = y + float(rng.uniform(-0.008, 0.008))
        return (jx, jy)
    return None


# 目标位置采样带（base 系，经验可达区，来自扳手可达扫描）：(x_lo,x_hi,y_lo,y_hi,weight)
# 主带 = 高成功率可靠区；扩展带 = 更广位置覆盖，失败由运行时校验+重试兜底。
TARGET_BANDS = [
    (0.50, 0.60, -0.50, -0.12, 0.75),
    (0.44, 0.68, -0.55, -0.10, 0.25),
]


def _sample_target_xy(rng, guard):
    """目标位置：连续均匀采样于可达带（避开工具箱）。返回 (x,y) 或 None。"""
    ws = np.asarray([b[4] for b in TARGET_BANDS], dtype=np.float64)
    ws = ws / ws.sum()
    for _ in range(60):
        b = TARGET_BANDS[int(rng.choice(len(TARGET_BANDS), p=ws))]
        x_lo, x_hi, y_lo, y_hi = b[:4]
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        if guard is None or not _cell_in_box((x, y), guard):
            return (round(x, 4), round(y, 4))
    return None


def _sample_episode_layout(rng, cells, guard, tool_present_pool):
    """返回 (target_idx, present:list[int], xy:dict[int,(x,y)])。

    目标放可达带内任意连续位置；干扰物在稳定格点池上尽量铺开；
    找不到足够位置时减少干扰物数量(仍保证 k≥1)。
    """
    target = int(rng.integers(0, len(tool_present_pool)))
    # 先定目标，再定 k
    max_k = 5
    if target in (KNIFE_L_IDX, KNIFE_R_IDX):
        max_k = 4  # 目标是一把电工刀时，另一把电工刀不在场（视觉不可区分）
    # 干扰物池：除目标外；若目标是刀，剔除另一把刀
    distractor_pool = [t for t in range(5) if t != target]
    if target in (KNIFE_L_IDX, KNIFE_R_IDX):
        twin = KNIFE_R_IDX if target == KNIFE_L_IDX else KNIFE_L_IDX
        distractor_pool = [t for t in distractor_pool if t != twin]
    k = int(rng.integers(1, max_k + 1))
    distractors = [int(t) for t in rng.choice(distractor_pool, size=min(k - 1, len(distractor_pool)), replace=False)]
    present = [target] + distractors

    xy: dict[int, tuple[float, float]] = {}
    occupied: list[tuple[float, float]] = []
    # 目标优先放（可达带内连续位置）
    target_xy = _sample_target_xy(rng, guard)
    if target_xy is None:
        return None
    xy[target] = target_xy
    occupied.append(target_xy)

    # 干扰物依次铺开
    min_sep = 0.17
    for t in distractors:
        c = None
        for _ in range(40):
            c = _pick_cell(rng, cells, occupied, min_sep)
            if c is not None and not _cell_in_box(c, guard):
                break
            c = None
        if c is not None:
            xy[t] = c
            occupied.append(c)
    present = [t for t in present if t in xy]
    return target, present, xy


def _place_scene_tools(env, base_body: str, present: list[int], xy: dict[int, tuple[float, float]]):
    """设置全部 5 件工具 free joint：在场→指定格点；不在场→桌下掉落区。

    姿态保留各工具参考四元数（纯平移）。
    """
    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    present_set = set(present)
    stash_y_iter = iter(_STASH_YS)
    qpos = {}
    for tool_idx in range(5):
        if tool_idx in present_set:
            x, y = xy[tool_idx]
            pos_b = np.array([x, y, float(TOOL_REF_POS_B[tool_idx, 2])], dtype=np.float64)
        else:
            try:
                sy = next(stash_y_iter)
            except StopIteration:
                sy = -0.60
            pos_b = np.array([_STASH_X, sy, float(TOOL_REF_POS_B[tool_idx, 2])], dtype=np.float64)
        rot_b = R.from_quat(np.asarray(TOOL_REF_QUAT_XYZW_B[tool_idx], dtype=np.float64))
        world_pos = base_pos + base_rot.apply(pos_b)
        world_quat_xyzw = (base_rot * rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        qpos[TOOL_JOINTS[tool_idx]] = np.concatenate([world_pos, world_quat_wxyz])
    env.set_joint_qpos(qpos)
    env.mj_forward()


def _settle(env, mgr, steps: int):
    """不开采集地推进物理，让工具落地静止（机器人保持默认位姿附近）。"""
    if steps <= 0:
        return
    try:
        mgr.set_init_ctrl()
        env.set_ctrl(mgr.ctrl)
        for _ in range(int(steps)):
            env.step(mgr.ctrl)
    except Exception as e:
        orca_logger.warning(f"[settle] 物理沉降异常: {e}")


def _query_pos_b(env, base_body: str) -> np.ndarray:
    return _T._query_tool_positions_b(env, base_body)


def _grasp_lift_segments(wps, safe_z: float, f: dict, ctrl_steps):
    """生成单工具 抓取+抬升+保持 的分段（每段时长以帧计，经 ctrl_steps 换算）。"""
    wp0, wp1 = wps[0], wps[1]

    def above(p, z):
        return [p[0], p[1], z]

    def seg(frames, target_b, quat, grip, label):
        return {
            "steps": ctrl_steps(int(frames)),
            "l_hold": True,
            "r_target_b": target_b,
            "r_quat_b": quat,
            "gripper_r": grip,
            "label": label,
        }

    return [
        seg(f["transit"], above(wp0["pos"], safe_z), wp0["quat"], "open", "G1-高位过渡"),
        seg(f["descend"], wp0["pos"], wp0["quat"], "open", "G2-下降接近"),
        seg(f["grasp"], wp1["pos"], wp1["quat"], "open", "G3-移到抓取点"),
        seg(f["settle"], wp1["pos"], wp1["quat"], "close", "G4-闭爪驻留"),
        seg(f["lift"], above(wp1["pos"], safe_z), wp1["quat"], "close", "G5-抬升"),
        seg(f["hold"], above(wp1["pos"], safe_z), wp1["quat"], "close", "G6-保持"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 单工具任意位姿抓取脚本化采集 → LeRobot v2.1")
    parser.add_argument("--task_config", type=str, default="../common/example.yaml")
    parser.add_argument(
        "--lerobot_out",
        type=str,
        default=os.path.join(OPENPI_DATA_ROOT, "g1_omnipicker_single_grasp"),
    )
    parser.add_argument("--repo_id", default="local/g1_omnipicker_single_grasp")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_attempts", type=int, default=6, help="单集成功前的最大尝试次数")
    parser.add_argument("--placeable_scan", type=str, default="/tmp/placeable_scan.json")
    parser.add_argument("--safe_z", type=float, default=0.48, help="抓取后保持高度(base 系 z)")
    parser.add_argument("--settle_steps", type=int, default=180, help="放工具后的物理沉降步数")
    parser.add_argument(
        "--smoke", type=int, default=0,
        help=">0: 冒烟模式，每 ep 开始时把 head/wrist 帧存到 /tmp/grasp_debug/epN_*.png，共 smoke 集",
    )
    parser.add_argument(
        "--waypoint_files",
        type=str,
        default=",".join(os.path.join(base_dir, f"my_waypoint_tool{i}.yaml") for i in range(1, 6)),
    )
    parser.add_argument(
        "--x_range", type=str, default="0.42,0.80", help="允许放置 x 区间(逗号分隔)"
    )
    parser.add_argument(
        "--y_range", type=str, default="-0.60,-0.07", help="允许放置 y 区间(逗号分隔)"
    )
    # 各分段帧数
    parser.add_argument("--f_transit", type=int, default=45)
    parser.add_argument("--f_descend", type=int, default=25)
    parser.add_argument("--f_grasp", type=int, default=30)
    parser.add_argument("--f_settle", type=int, default=28)
    parser.add_argument("--f_lift", type=int, default=30)
    parser.add_argument("--f_hold", type=int, default=25)
    parser.add_argument("--kp", type=float, default=220.0)
    parser.add_argument(
        "--fixed_layouts",
        type=str,
        default="",
        help="逗号分隔的 FIXED_SMOKE_LAYOUTS 索引；先用完这些再随机",
    )
    args = parser.parse_args()

    fixed_queue: list[tuple] = []
    if args.fixed_layouts.strip():
        for s in args.fixed_layouts.split(","):
            s = s.strip()
            if s == "":
                continue
            idx = int(s)
            if 0 <= idx < len(FIXED_SMOKE_LAYOUTS):
                fixed_queue.append(FIXED_SMOKE_LAYOUTS[idx])
        orca_logger.info(f"[布局] 固定布局队列: {len(fixed_queue)} 个，随后进入随机采样")

    x_lo, x_hi = (float(v) for v in args.x_range.split(","))
    y_lo, y_hi = (float(v) for v in args.y_range.split(","))

    cells = _load_placeable_cells(args.placeable_scan, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi)
    if not cells:
        orca_logger.error("没有可用放置网格（检查 --placeable_scan）")
        return

    # ── 路点 ─────────────────────────────────────────────
    tool_data: list[dict] = []
    g_open_global = -0.8561
    g_close_global = 2.0
    for i, path in enumerate([p.strip() for p in args.waypoint_files.split(",")]):
        if not os.path.exists(path):
            orca_logger.error(f"路点文件不存在: {path}")
            return
        td = _T._load_waypoint_yaml(os.path.abspath(path))
        tool_data.append(td)
        if i == 0:
            g_open_global = td["g_open"]
            g_close_global = td["g_close"]

    # ── 环境/控制器/相机（同工具采集接线）──────────────
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    _n_motor = len(agent_conf.gripper_l["actuator_names"]) + len(agent_conf.gripper_r["actuator_names"])

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_single_grasp")
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    env_dt = float(manager.time_step) * int(manager.frame_skip)
    frame_dt = 1.0 / float(args.fps)
    steps_scale = max(1, int(round(frame_dt / env_dt)))
    scaled_dt = steps_scale * env_dt
    orca_logger.info(f"[时序] steps_scale={steps_scale} (env_dt={env_dt:.4f}, frame_dt={frame_dt:.4f})")
    if abs(scaled_dt - frame_dt) > 1e-6:
        orca_logger.warning("轨迹控制周期与采集帧率存在偏差，请检查时序")

    def _ctrl_steps(frames: int) -> int:
        return max(1, int(round(float(frames) * steps_scale)))

    print("=" * 64, flush=True)
    print("  G1 OmniPicker · 任意位置单工具抓取采集", flush=True)
    print(f"  目标: {args.num_episodes} 集 | 网格 {len(cells)} 个 | safe_z={args.safe_z}", flush=True)
    print(f"  输出: {args.lerobot_out}", flush=True)
    print("=" * 64, flush=True)

    # ── 相机与数据集 ─────────────────────────────────────
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = dict(DEFAULT_CAMERA_MAP)
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")
    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return
    cam_shape = (3, cam_hw[0], cam_hw[1])
    print(f"  相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}", flush=True)

    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=args.lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_omnipicker",
    )
    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task="抓取扳手",  # 占位，每集 set_task 覆盖
        clock="sim",
    )

    # ── 控制器 ───────────────────────────────────────────
    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        orca_logger.error("场景初始化失败")
        env.close()
        return
    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])
    base_body = env.body(agent_conf.base_body)

    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)
    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)

    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    guard = _box_guard_xy_minmax(env, base_body)
    if guard is not None:
        _kept = [c for c in cells if not _cell_in_box(c, guard)]
        if len(_kept) < 10:
            orca_logger.warning(f"[布局] 排除工具箱后仅剩 {len(_kept)} 个格点，保留全部")
        else:
            cells = _kept
            orca_logger.info(f"[布局] 排除工具箱占用后可用格点: {len(cells)}")

    rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))

    start_saved = writer.num_episodes
    n_success = 0
    stats = {"attempt": 0, "success": 0, "fail": {"grasp": 0, "layout": 0, "other": 0, "giveup": 0}}
    smoke_budget = max(0, int(args.smoke))
    if smoke_budget:
        os.makedirs("/tmp/grasp_debug", exist_ok=True)
    else:
        smoke_budget = -1  # 非冒烟 = 不存帧

    try:
        with writer:
            attempts_since_save = 0
            while n_success < args.num_episodes:
                attempts_since_save += 1
                if attempts_since_save > args.max_attempts:
                    # 单集连续失败上限：放弃本集，重开新一轮，避免不可达布局无限重试
                    stats["fail"]["giveup"] += 1
                    attempts_since_save = 0
                    storage.clear_data()
                    orca_logger.warning(
                        f"[放弃] 连续 {args.max_attempts} 次未保存，重开新一轮（已存 {n_success}）"
                    )
                    print(f"  [放弃] 连续 {args.max_attempts} 次失败，跳到下一轮", flush=True)
                    continue
                stats["attempt"] += 1
                attempt_no = stats["attempt"]
                ep_display = start_saved + n_success + 1
                if fixed_queue:
                    layout = fixed_queue.pop(0)
                    target = layout[0]
                    xy = dict(layout[1])
                    present = list(xy.keys())
                else:
                    layout = _sample_episode_layout(rng, cells, guard, list(range(5)))
                    if layout is None:
                        orca_logger.warning("[布局] 无法采到目标格点，重试")
                        stats["fail"]["layout"] += 1
                        continue
                    target, present, xy = layout

                env.reset()
                time.sleep(0.05)
                if not manager.update_scene():
                    orca_logger.info("场景更新失败，停止采集")
                    break
                env.set_default_joint_values(default_joint_values)
                base_body = env.body(agent_conf.base_body)

                _place_scene_tools(env, base_body, present, xy)
                _settle(env, manager, args.settle_steps)
                env.set_default_joint_values(default_joint_values)  # 重新归位机器人
                env.mj_forward()

                actual = _query_pos_b(env, base_body)
                z_before = float(actual[target, 2])
                delta_b = actual - TOOL_REF_POS_B

                # 运行时布局校验（基于静置后真实位置，能挡住扫描表外的边缘/翘边格点）
                present_z = [float(actual[t, 2]) for t in present]
                layout_bad = False
                if not (0.118 <= z_before <= 0.20):
                    orca_logger.warning(f"[布局] 目标静置高度异常 z={z_before:.3f}，重采")
                    layout_bad = True
                elif any(z < 0.095 for z in present_z):
                    orca_logger.warning(
                        f"[布局] 有工具落桌下 z={[round(z,3) for z in present_z]}，重采"
                    )
                    layout_bad = True
                if layout_bad:
                    stats["fail"]["layout"] += 1
                    continue
                # 用实际落点算在场间距
                if len(present) >= 2:
                    coords = actual[np.array(present), :2]
                    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                    np.fill_diagonal(d, np.inf)
                    if d.min() < 0.11:
                        orca_logger.warning(f"[布局] 在场工具实际过近，重采 (min={d.min():.3f})")
                        stats["fail"]["layout"] += 1
                        continue

                prompt = _tool_prompt(target, rng)
                storage.set_task(prompt)

                # 目标路点：只平移 wp0/wp1（接近/抓取点）到实际落点
                td = tool_data[target]
                warped = _T._warp_waypoints(td, delta_b[target])
                segs = _grasp_lift_segments(
                    warped["waypoints"],
                    safe_z=float(args.safe_z),
                    f={
                        "transit": args.f_transit,
                        "descend": args.f_descend,
                        "grasp": args.f_grasp,
                        "settle": args.f_settle,
                        "lift": args.f_lift,
                        "hold": args.f_hold,
                    },
                    ctrl_steps=_ctrl_steps,
                )
                all_segments = list(segs)

                layout_txt = " | ".join(f"{TOOL_NAMES[t]}@({xy[t][0]:.2f},{xy[t][1]:.2f})" for t in present)
                print(f"\n>>> [尝试{attempt_no}] 抓取 {TOOL_NAMES[target]} | 在场 {len(present)}: {layout_txt}", flush=True)

                # 冒烟帧（记录前静态场景）
                dump_post = False
                if smoke_budget > 0:
                    dump_post = True
                    smoke_budget -= 1
                    try:
                        from dataStorage.lerobot_camera import capture_frame_with_idx
                        imgs, _ = capture_frame_with_idx(cameras, camera_map, cam_hw)
                        for k, arr in imgs.items():
                            from PIL import Image
                            Image.fromarray(arr).save(f"/tmp/grasp_debug/ep{ep_display}_pre_{k}.png")
                        print(f"  [冒烟] pre 帧已存 ep{ep_display}", flush=True)
                    except Exception as e:
                        orca_logger.warning(f"[冒烟] 保存调试帧失败: {e}")

                # 预构建轨迹并运行
                try:
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = scripted.build_segmented_trajectory(
                        env, agent_conf, all_segments, g_open_global, g_close_global
                    )
                except Exception as e:
                    orca_logger.error(f"[轨迹] 构建失败: {e}")
                    stats["fail"]["other"] += 1
                    continue

                device = _T.G1ScriptedTrajectoryDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                    place_arm_at=None, box_bottom=None, base_body_query=None,
                )
                manager.set_device(device)
                manager.run_episode()

                if dump_post:
                    try:
                        from dataStorage.lerobot_camera import capture_frame_with_idx
                        imgs, _ = capture_frame_with_idx(cameras, camera_map, cam_hw)
                        for k, arr in imgs.items():
                            from PIL import Image
                            Image.fromarray(arr).save(f"/tmp/grasp_debug/ep{ep_display}_post_{k}.png")
                        print(f"  [冒烟] post 帧已存 ep{ep_display}", flush=True)
                    except Exception as e:
                        orca_logger.warning(f"[冒烟] 保存 post 帧失败: {e}")

                actual_after = _query_pos_b(env, base_body)
                z_after = float(actual_after[target, 2])
                grasped = z_after > max(_GRASP_OK_Z, z_before + 0.12)

                if not grasped:
                    stats["fail"]["grasp"] += 1
                    orca_logger.warning(
                        f"[✗] 第{attempt_no}次未抓住 {TOOL_NAMES[target]} "
                        f"(z {z_before:.3f}→{z_after:.3f})，清缓存重采"
                    )
                    print(f"  [✗] 未抓住 (z {z_before:.3f}→{z_after:.3f})", flush=True)
                    storage.clear_data()
                    continue

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                stats["success"] += 1
                attempts_since_save = 0
                print(f">>> [✓] 第 {ep_display} 集已保存（{TOOL_NAMES[target]}，{len(present)} 把在场）", flush=True)
                orca_logger.info(
                    f"[✓] Episode {ep_display} 保存成功 (尝试{attempt_no})"
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = (
            f"采集结束: 成功 {stats['success']} 集 / 尝试 {stats['attempt']} "
            f"(grasp失败 {stats['fail']['grasp']}, layout {stats['fail']['layout']}, "
            f"giveup {stats['fail']['giveup']}, other {stats['fail']['other']}); "
            f"数据集 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        )
        orca_logger.info(summary)
        print(f"\n{'=' * 64}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {args.lerobot_out}", flush=True)
        print("=" * 64, flush=True)
        try:
            scene_manager.show_ui_message(1, "", showtime=0)
            env.render()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("已收到中断请求")
    except Exception as e:
        orca_logger.error(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        orca_logger.info("程序已退出")
        os._exit(0)
