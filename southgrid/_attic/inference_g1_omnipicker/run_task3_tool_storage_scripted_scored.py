"""Task3 电力工具归纳收置 · 脚本航点轨迹评分驱动（不依赖学习策略）

与 run_task3_tool_storage_scored.py（策略版）的区别：本脚本完全不连策略服务，
直接复用官方 scripted 采集程序的仿真接口与航点轨迹，用几何运动完成 5 件工具的
抓取—入箱，再由 orca_scorer_client 评分并上传中央服务器。

复用来源：examples/dataCollection/g1_omnipicker/
            g1_omnipicker_collection_scripted_tool_lerobot.py

与采集版的差异（都是为"单次评分"服务）：
  1. 不做随机化、不做 --place_tools 复位：工具保持在官方 Task3 场景的原始位置。
  2. 按实测工具位与 _TOOL_REFERENCE_POS_B 的偏差对航点做整体平移（warp），
     使轨迹贴合当前场景，而不是假设两者完全对齐。
  3. 传入 box_bottom=None 给设备 → 关闭"入箱失败即 force_task_end"的监视。
     评分按每件工具是否入箱单独计分，中途中止会白丢后面工具的分，必须跑满。
  4. 不写 LeRobot 数据集、不起相机（策略才需要图像）。
  5. 软时限看门狗：到点置 _shutdown_requested，保证在 180s 硬限内收尾并 finish。

用法（务必用 orcalab 的绝对路径 python）：
    # 只读预演：构建轨迹、打印总步数与预估墙钟，不评分不动手
    .../bin/python run_task3_tool_storage_scripted_scored.py --dry-run

    # 正式评分
    .../bin/python run_task3_tool_storage_scripted_scored.py \
        --speed 1.6 --team-id csgComp_Team_08 --team-token <你的队伍token>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
_dc_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "dataCollection", "g1_omnipicker"))
for _p in (base_dir, _dc_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g1_omnipicker_collection_scripted_tool_lerobot as S  # noqa: E402
from dataCollectionManager.data_collection_manager import DataCollectionManager  # noqa: E402
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
from task.abstract_task import EmptyTask  # noqa: E402
from controllers.controller_task import TaskStatusController  # noqa: E402
from controllers.controllers import (  # noqa: E402
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from controllers.controller_2f85_reverse import Controller2F85Reverse  # noqa: E402
from conf import g1_omnipicker_conf as agent_conf  # noqa: E402
from yaml import Loader, load, safe_load  # noqa: E402

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

# 评分口径（与 configs/tasks.yaml / 策略版驱动一致）
TOOLBOX_SITE = "Group_Interactive_ToolBox_base_site"
SCORE_SITES = [
    ("pick_wrench",            "Group_Interactive_Spanner_task_spanner_site"),
    ("pick_screwdriver",       "Group_Interactive_Screwdriver_task_screwdriver_site"),
    ("pick_electrician_knife", "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_site"),
    ("pick_flashlight",        "Group_Interactive_Flashlight_task_flashlight_site"),
    ("pick_voltage_pen",       "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_site"),
]
PLACED_DIST = 0.20          # 工具 site 距 toolbox site 小于此值视为已入箱
ROBOT_BASE_BODY = "g1_omnipicker_body_link1"


def _site_pos(env, name: str):
    try:
        d = env.query_site_pos_and_quat([name])
        return np.asarray(d[name]["xpos"], dtype=np.float64).reshape(3)
    except Exception:
        return None


def _body_pos(env, name: str):
    try:
        xpos, _, _ = env.get_body_xpos_xmat_xquat([name])
        return np.asarray(xpos, dtype=np.float64).reshape(3)
    except Exception:
        return None


def _dist(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _finish_with_retry(scorer, tries: int = 6, gap: float = 20.0) -> dict:
    """调用 scorer.finish() 直到拿到真实分数。

    服务端 finish_attempt 是同步的（回溯评分→惩罚→停录屏→建摘要→上报中央），
    停一个上百秒的录屏可能远超 HTTP 超时。超时后 _post 会吞掉异常返回 error 字典，
    但服务端那次 finish 仍在继续；重试时 _finishing/_last_summary 会让它直接返回
    已算好的摘要，所以重试是安全且幂等的。
    """
    last: dict = {}
    for i in range(max(1, tries)):
        s = scorer.finish()
        if isinstance(s, dict) and s.get("score") is not None:
            return s
        last = s if isinstance(s, dict) else {}
        _log(f"[scorer] finish 未拿到分（{last}），{gap:.0f}s 后重试 {i + 1}/{tries}")
        time.sleep(gap)
    return last


def _scan_placed(env, box) -> dict:
    """当前 5 件工具各自的入箱状态。"""
    out = {}
    for step_id, site in SCORE_SITES:
        d = _dist(_site_pos(env, site), box)
        out[step_id] = (d <= PLACED_DIST, d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Task3 脚本航点轨迹评分接入（无策略）")
    ap.add_argument("--level", type=str, default="default")
    ap.add_argument("--task_config", type=str,
                    default=os.path.join(_dc_dir, "..", "common", "example.yaml"))
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--team-token", default=None)
    ap.add_argument("--robot-id", default="g1_omnipicker",
                    help="注意：评分配置文件里是过期的 g1_pick，此处显式覆盖")
    ap.add_argument("--waypoint_files", type=str,
                    default=",".join(os.path.join(_dc_dir, f"my_waypoint_tool{i}.yaml") for i in range(1, 6)))
    ap.add_argument("--extra_slot_waypoints", type=str,
                    default=os.path.join(_dc_dir, "my_slot_waypoints.yaml"))
    ap.add_argument("--safe_z", type=float, default=0.50)
    ap.add_argument("--speed", type=float, default=1.6,
                    help="轨迹提速倍率。实测 1.0x 全程约 227s 墙钟，超过 180s 硬限，必须 >1.4")
    ap.add_argument("--kp", type=float, default=220.0)
    ap.add_argument("--no_warp", action="store_true",
                    help="不做工具位平移补偿，直接用原始航点（默认做 warp）")
    ap.add_argument("--no_restore_tools", action="store_true",
                    help="不复位工具（默认：若发现工具已在箱内，恢复到官方参考位姿，"
                         "使 attempt 起始状态与全新场景一致）")
    ap.add_argument("--soft_deadline_s", type=float, default=166.0,
                    help="软时限，到点强制收尾（本地 scorer 硬限 180s）")
    ap.add_argument("--dry-run", action="store_true", help="只构建轨迹+测速，不评分")
    ap.add_argument("--no-score", action="store_true", help="跑动作但不评分")
    ap.add_argument("--calib_steps", type=int, default=80, help="dry-run 实测步速用的步数")
    # 轨迹时长参数（帧，1/fps 秒），与采集版默认一致
    for k, v in dict(steps_transit=50, steps_descend=30, steps_grasp=35, steps_settle=30,
                     steps_lift=35, steps_place_via=45, steps_to_box=50, steps_release=45,
                     steps_release_settle=20, steps_lift_after=15).items():
        ap.add_argument(f"--{k}", type=int, default=v)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--jsonl", type=str, default="")
    ap.add_argument("--scorer_timeout", type=float, default=300.0,
                    help="ScorerClient HTTP 超时(秒)。默认 5s 会在服务端停录屏+回溯评分时超时，"
                         "返回 error 字典导致拿不到分；必须给足")
    ap.add_argument("--upload_wait_s", type=float, default=90.0,
                    help="finish 后等待视频上传的秒数（本地服务随父进程退出，不能提前杀）")
    args = ap.parse_args()

    # ── 载入 5 个航点 YAML ──────────────────────────────────────────────────
    yaml_paths = [p.strip() for p in args.waypoint_files.split(",")]
    if len(yaml_paths) != 5:
        _log(f"[setup] --waypoint_files 需要 5 个路径，收到 {len(yaml_paths)}")
        return
    tool_data, g_open, g_close = [], -0.8561, 2.0
    for i, p in enumerate(yaml_paths):
        td = S._load_waypoint_yaml(os.path.abspath(os.path.expanduser(p)))
        tool_data.append(td)
        if i == 0:
            g_open, g_close = td["g_open"], td["g_close"]
    _log(f"[setup] 已载入 5 件工具航点（各 {len(tool_data[0]['waypoints'])} 点）")

    extra_slot_waypoints = {}
    if args.extra_slot_waypoints and os.path.isfile(args.extra_slot_waypoints):
        raw = S._load_extra_slot_waypoints(args.extra_slot_waypoints)
        for key, entry in raw.items():
            try:
                slot_idx = int(key.split("_")[0][4:])
            except (IndexError, ValueError):
                continue
            slot_y = float(S._TOOL_REFERENCE_POS_B[slot_idx, 1])
            if all(abs(float(entry[w]["pos"][1]) - slot_y) <= S._EXTRA_SLOT_Y_TOL for w in ("wp0", "wp1")):
                extra_slot_waypoints[key] = entry
        _log(f"[setup] 槽位路点 有效 {len(extra_slot_waypoints)}/{len(raw)} 条")

    # ── 场景与控制器 ────────────────────────────────────────────────────────
    with open(os.path.abspath(args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    default_joint_values = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], S._L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    scratch_dir = os.path.join(base_dir, "_scripted_scratch", args.level)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)
    _n_motor = len(agent_conf.gripper_l["actuator_names"]) + len(agent_conf.gripper_r["actuator_names"])

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    manager = DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=ENTRY_POINT,
        default_joint_values={}, obs_callback=_obs_callback_safe, env_index=0, device=None,
        scene_manager=scene_manager, data_storage=None, frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE

    env_dt = float(manager.time_step) * int(manager.frame_skip)
    steps_scale = max(1, int(round((1.0 / float(args.fps)) / env_dt)))

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        _log("[setup] 场景更新失败，退出")
        env.close()
        return
    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    base_body = env.body(agent_conf.base_body)

    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name,
                                      {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])})
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name,
                                      {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])})
    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _a in (l_arm, r_arm):
        _a.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _a.controller.kd = 2.0 * np.sqrt(_a.controller.kp)

    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname,
        {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])},
        Controller2F85Reverse.ControllerType.DATA)
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname,
        {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])},
        Controller2F85Reverse.ControllerType.DATA)

    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # ── 只读场景探测 ────────────────────────────────────────────────────────
    box = _site_pos(env, TOOLBOX_SITE)
    rbase = _body_pos(env, ROBOT_BASE_BODY)
    _log("\n[probe] === 当前 ORCA 场景 ===")
    _log(f"[probe] toolbox site : {None if box is None else box.round(3).tolist()}")
    _log(f"[probe] robot base   : {None if rbase is None else rbase.round(3).tolist()}")
    if box is None:
        _log("[probe] 找不到 toolbox site —— 当前不是官方 Task3 场景，无法评分")
        env.close()
        return

    # ── 工具复位 ────────────────────────────────────────────────────────────
    # 工具是场景里的 free-joint 物体，env.reset() / update_scene() 都不会复位它们。
    # 上一次演练或 attempt 会把工具留在箱内，此时直接开跑等于对着空货架抓 5 次。
    # 这里把 5 件工具恢复到官方参考位姿，使本次 attempt 的起始状态与全新场景一致。
    if not args.no_restore_tools:
        st0 = _scan_placed(env, box)
        stray = [k for k, v in st0.items() if v[0]]
        if stray:
            _log(f"[reset] 发现 {len(stray)} 件工具已在箱内 {stray}")
            _log("[reset] 恢复到官方参考位姿（等价于全新场景的初始布局）")
            S._place_tools(env, base_body, np.arange(5, dtype=np.int64))
            env.mj_forward()
            time.sleep(0.3)
            st1 = _scan_placed(env, box)
            _log("[reset] 复位后: " + " ".join(
                f"{k}={'IN' if v[0] else f'{v[1]:.2f}m'}" for k, v in st1.items()))
        else:
            _log("[reset] 工具均在箱外，无需复位")

    try:
        actual_b = S._query_tool_positions_b(env, base_body)
        delta_b = actual_b - S._TOOL_REFERENCE_POS_B
        _log("[probe] 工具位(base系) 实测 vs 参考 偏差:")
        for i in range(5):
            _log(f"[probe]   {S._TOOL_NAMES[i]:<10} d=({delta_b[i][0]:+.3f},{delta_b[i][1]:+.3f},"
                 f"{delta_b[i][2]:+.3f}) |d|={np.linalg.norm(delta_b[i]):.3f}")
    except Exception as e:
        _log(f"[probe] 工具位查询失败({e})，改用零偏差")
        actual_b, delta_b = None, np.zeros((5, 3))

    before = _scan_placed(env, box)
    _log("[probe] 初始入箱状态: " + " ".join(
        f"{k}={'IN' if v[0] else f'{v[1]:.2f}m'}" for k, v in before.items()))

    # ── 构建 5 件工具轨迹（固定顺序 0..4）───────────────────────────────────
    episode_tool_data = [S._copy_tool_spec(td) for td in tool_data]
    if not args.no_warp:
        for i in range(5):
            episode_tool_data[i] = S._warp_waypoints(episode_tool_data[i], delta_b[i])
        _log("[build] 已按实测偏差对 5 件工具航点做平移补偿 (warp)")
    for tool_idx in range(5):
        episode_tool_data[tool_idx], used = S._apply_extra_slot_waypoints(
            episode_tool_data[tool_idx], tool_idx, tool_idx, extra_slot_waypoints)
        if used:
            _log(f"[build] {S._TOOL_NAMES[tool_idx]} 使用槽位专用路径 slot{tool_idx}")

    # 槽位 5（最右）抓取位姿补偿
    t5 = S._SLOT5_IDX
    episode_tool_data[t5] = S._bias_grasp_waypoints_y(episode_tool_data[t5], S._SLOT5_GRASP_DY)
    # 左侧电工刀补偿
    episode_tool_data[S._KNIFE_L_IDX] = S._bias_waypoints(
        episode_tool_data[S._KNIFE_L_IDX], dy=S._TOOL3_GRASP_DY, dz=S._TOOL3_GRASP_DZ, wp_indices=(0, 1))
    _log("[build] 已应用 槽位5 / 左侧电工刀 抓取补偿")

    traj_speed = max(0.1, float(args.speed))

    def _ctrl_steps(frames: int) -> int:
        return max(1, int(round(float(frames) * steps_scale / traj_speed)))

    all_segments: list[dict] = []
    cum = 0
    for tool_idx in range(5):
        td = episode_tool_data[tool_idx]
        segs = S._build_tool_segments(
            wps=td["waypoints"], safe_z=args.safe_z,
            steps_transit=_ctrl_steps(args.steps_transit),
            steps_descend=_ctrl_steps(args.steps_descend),
            steps_grasp=_ctrl_steps(args.steps_grasp),
            steps_settle=_ctrl_steps(args.steps_settle),
            steps_lift=_ctrl_steps(args.steps_lift),
            steps_place_via=_ctrl_steps(args.steps_place_via),
            steps_to_box=_ctrl_steps(args.steps_to_box),
            steps_release=_ctrl_steps(args.steps_release),
            steps_release_settle=_ctrl_steps(args.steps_release_settle),
            steps_lift_after=_ctrl_steps(args.steps_lift_after),
        )
        for s in segs:
            s["label"] = f"工具{tool_idx + 1}({S._TOOL_NAMES[tool_idx]})-{s['label']}"
        total = sum(int(s["steps"]) for s in segs)
        _log(f"[build] {S._TOOL_NAMES[tool_idx]:<10} {len(segs)} 段 / {total} 步")
        all_segments.extend(segs)
        cum += total

    total_steps = cum
    _log(f"[build] 合计 {len(all_segments)} 段 / {total_steps} 步 / "
         f"仿真时长 {total_steps * env_dt:.1f}s (speed={traj_speed:.2f}x)")

    l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = S.scripted.build_segmented_trajectory(
        env, agent_conf, all_segments, g_open, g_close)
    _log(f"[run] 轨迹已预构建：{len(l_pos)} 步")

    # box_bottom=None → 关闭"入箱失败即中止"，5 件工具必须全部跑完
    device = S.G1ScriptedTrajectoryDevice(
        l_arm, r_arm, l_grip, r_grip, task_status,
        l_pos, l_quat, r_pos, r_quat, l_gm, r_gm,
        place_arm_at=None, box_bottom=None, base_body_query=base_body,
    )
    manager.set_device(device)

    if args.dry_run:
        # 带真实 scripted device 实测控制步墙钟（比空转更接近实际）
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        env.mj_forward()
        for c in manager.controllers:
            c.reset()
        t0 = time.time(); N = max(1, int(args.calib_steps))
        for _ in range(N):
            act = manager.run_controllers()
            env.step(act); env.render()
        dt = (time.time() - t0) / N
        _log(f"\n[dry-run] 控制步均值 {dt * 1000:.1f} ms → 全程 {total_steps} 步预估 "
             f"{total_steps * dt:.0f}s 墙钟 (软限 {args.soft_deadline_s:.0f}s / 硬限 180s)")
        if total_steps * dt > 180:
            _log(f"[dry-run] ⚠ 预估超硬限，建议 --speed "
                 f"{traj_speed * total_steps * dt / 160.0:.2f} 或更大")
        else:
            _log("[dry-run] ✓ 预估在硬限内")
        env.close()
        return

    # ── 评分 attempt ────────────────────────────────────────────────────────
    scorer = None
    if not args.no_score:
        from orca_scorer_client import ScorerClient
        scorer = ScorerClient(team_id=args.team_id, team_token=args.team_token,
                              robot_id=args.robot_id, timeout=args.scorer_timeout)
        info = scorer.start_attempt("task3_tool_storage", prompt="整理工具(脚本航点轨迹)")
        _log(f"[scorer] attempt started: {info.get('attempt_id')} robot_id={args.robot_id}")
    else:
        _log("[scorer] 演练模式（不评分）")

    t_start = time.time()
    stop_flag = {"hit": False}

    def _watchdog():
        while not stop_flag["hit"]:
            if time.time() - t_start >= args.soft_deadline_s:
                _log(f"\n[watchdog] 软时限 {args.soft_deadline_s:.0f}s 到，强制收尾")
                try:
                    manager._shutdown_requested = True
                except Exception:
                    pass
                return
            time.sleep(0.5)

    def _progress():
        # 只读 device.t 这个纯计数，绝不在此线程里 query env —— 会与主线程的
        # env.step 抢底层模型状态。入箱结果在主循环结束后统一扫描。
        while not stop_flag["hit"]:
            time.sleep(5.0)
            if stop_flag["hit"]:
                return
            _log(f"    [t={time.time() - t_start:5.1f}s] 轨迹步 {device.t}/{total_steps}")

    wd = threading.Thread(target=_watchdog, daemon=True)
    pg = threading.Thread(target=_progress, daemon=True)
    wd.start(); pg.start()

    finished = False
    try:
        manager.run_episode()
        stop_flag["hit"] = True
        _log(f"\n[run] 轨迹执行结束，用时 {time.time() - t_start:.1f}s，"
             f"步 {device.t}/{total_steps}，place_failed={device.place_failed}")

        after = _scan_placed(env, box)
        n_in = sum(1 for v in after.values() if v[0])
        _log(f"[run] 入箱结果 {n_in}/5:")
        for k, (ok, d) in after.items():
            _log(f"    {k:<26} {'✓ 已入箱' if ok else f'✗ 距箱 {d:.3f}m'}")

        if scorer:
            summary = _finish_with_retry(scorer)
            finished = True
            _log(f"\n[scorer] score: {summary.get('score')} / {summary.get('max_score')} "
                 f"(success_rate={summary.get('success_rate')}, status={summary.get('status')})")
            for r in summary.get("step_results", []):
                _log(f"  {r.get('step_id'):<26} {'PASS' if r.get('success') else 'FAIL'} "
                     f"score={r.get('score')} w={r.get('weight')}")
            _log(f"[scorer] 等待视频上传 {args.upload_wait_s:.0f}s …")
            time.sleep(args.upload_wait_s)
        if args.jsonl:
            Path(args.jsonl).write_text(json.dumps({
                "steps": int(device.t), "planned_steps": int(total_steps),
                "wall_s": round(time.time() - t_start, 1), "speed": traj_speed,
                "in_box": {k: bool(v[0]) for k, v in after.items()},
                "n_in_box": n_in,
            }, ensure_ascii=False), encoding="utf-8")
    finally:
        stop_flag["hit"] = True
        if scorer and not finished:
            try:
                summary = _finish_with_retry(scorer, tries=3)
                _log(f"\n[scorer] (收尾) score: {summary.get('score')} / {summary.get('max_score')}")
                for r in summary.get("step_results", []):
                    _log(f"  {r.get('step_id'):<26} {'PASS' if r.get('success') else 'FAIL'}")
                # 本地评分服务绑定了 --parent-pid，本进程一退它就被带走；
                # 必须留足时间给它把摘要上报中央、视频传完。
                _log(f"[scorer] (收尾) 等待上传 {args.upload_wait_s:.0f}s …")
                time.sleep(args.upload_wait_s)
            except Exception as fe:
                _log(f"[scorer] 收尾 finish 失败: {fe}")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("\n[task3-scripted] 中断")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)
