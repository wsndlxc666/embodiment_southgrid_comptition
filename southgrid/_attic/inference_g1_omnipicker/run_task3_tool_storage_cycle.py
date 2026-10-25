"""Task3 电力工具归纳收置 · 评分接入驱动（g1_omnipicker）

在单个评分 attempt（task3_tool_storage，本地 scorer 180s 硬时限）内，用 OpenPI
工具策略依次以训练 prompt 抓取 5 件工具放入工具箱；复用 eval 脚本的控制器/
相机/策略管线（左臂锁定，右臂 OSC）。由 orca_scorer_client 的 ScorerClient
自动评分（本地 scorer 采样 50051 场景 + 录屏上传中央服务器）。

工具是场景中的静态 free-joint 物体，env.reset()/update_scene() 不会复位它们，
因此不重置场景、只切换 prompt 连续收纳，物理全程连续。

用法（务必用 orcalab 的绝对路径 python，避免 openpi venv 抢占解释器）：
    # 0) 只读探测当前场景（不连相机/不评分/不动工具）
    .../bin/python run_task3_tool_storage_scored.py --dry-run

    # 1) 正式评分
    .../bin/python run_task3_tool_storage_scored.py \
        --team-id csgComp_Team_08 --team-token <你的队伍token>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

# ---------------------------------------------------------------------------
# Task3 评分常量（与 configs/tasks.yaml 一致）
# ---------------------------------------------------------------------------
TOOLBOX_SITE = "Group_Interactive_ToolBox_base_site"
SCORE_PHASES: list[dict] = [
    {"step_id": "pick_voltage_pen",       "site": "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_site", "prompt": "电工刀(右)"},
    {"step_id": "pick_screwdriver",       "site": "Group_Interactive_Screwdriver_task_screwdriver_site",                "prompt": "螺丝刀"},
    {"step_id": "pick_wrench",            "site": "Group_Interactive_Spanner_task_spanner_site",                        "prompt": "扳手"},
    {"step_id": "pick_flashlight",        "site": "Group_Interactive_Flashlight_task_flashlight_site",                  "prompt": "手电筒"},
    {"step_id": "pick_electrician_knife", "site": "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_site", "prompt": "电工刀(左)"},
]
ROBOT_BASE_BODY = "g1_omnipicker_body_link1"
# 右手指末端 site（用于诊断：区分"伸错位置 / 够不到 / 抓不住"）
EE_SITE_R = "g1_omnipicker_ee_center_site_r"
# 触发"已放入"的距离阈值（工具 site 距 toolbox site），m
PLACED_DIST = 0.20
# 右夹爪内爪电机目标值低于此视为张开/释放（-0.856 附近为张开，+值越大越闭合）
GRIP_OPEN_MOTOR = -0.35
# 释放确认保持时间
CONFIRM_S = 1.2
# 评分 attempt 软时限（本地 scorer 硬限 180s），预留 finish+上传
SOFT_DEADLINE_S = 165.0


def _site_pos(env, name: str) -> np.ndarray | None:
    try:
        d = env.query_site_pos_and_quat([name])
        return np.asarray(d[name]["xpos"], dtype=np.float64).reshape(3)
    except Exception:
        return None


def _body_pos(env, name: str) -> np.ndarray | None:
    try:
        xpos, _, _ = env.get_body_xpos_xmat_xquat([name])
        return np.asarray(xpos, dtype=np.float64).reshape(3)
    except Exception:
        return None


def _dist(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _grip_open(device) -> bool:
    """按右夹爪电机目标判断是否张开。"""
    ctrl = np.asarray(device.r_grip_ctrl, dtype=np.float64).reshape(-1)
    return float(ctrl[0]) < GRIP_OPEN_MOTOR


def _dump_frames(cameras: dict, tag: str, out_dir: str) -> None:
    """把当前各相机帧存成 PNG，供事后核对策略看到什么画面。"""
    if not cameras or not out_dir:
        return
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    for name, cam in cameras.items():
        try:
            f, _ = cam.get_frame(format="rgb24")
            if f is not None and f.size:
                p = os.path.join(out_dir, f"{tag}_{name}.png")
                cv2.imwrite(p, cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Task3 工具收纳评分接入")
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--team-token", default=None)
    ap.add_argument("--robot-id", default="g1_omnipicker")
    ap.add_argument("--dry-run", action="store_true", help="只读探测场景+测速，不评分不动工具")
    ap.add_argument("--no-score", action="store_true", help="跑动作但不评分（演练用）")
    ap.add_argument("--prompt", type=str, default="整理工具", help="整场单一指令（官方语气）")
    ap.add_argument("--max_steps", type=int, default=24000, help="整场最大控制步数")
    ap.add_argument("--per_tool_max_s", type=float, default=55.0, help="单件指令窗口秒数，超时未入箱则轮换到下一件")
    ap.add_argument("--action_repeat", type=int, default=10, help="每个动作目标连续执行步数(官方成功用 10)")
    ap.add_argument("--kp", type=float, default=220.0, help="OSC 阻抗刚度 kp(官方成功用 220)")
    ap.add_argument("--grasp_integral", action="store_true")
    ap.add_argument("--camera_warmup_steps", type=int, default=10)
    ap.add_argument("--smoke", type=int, default=0, help="连相机+策略后跑 N 步推理即退出(验证管线，不评分)")
    ap.add_argument("--per_phase_max_s", type=float, default=999.0, help="单工具最长墙钟秒(0=自动均分)")
    ap.add_argument("--only_phases", type=str, default="", help="逗号分隔的 step_id 子集(如 pick_wrench)，默认全部")
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--jsonl", type=str, default="")
    ap.add_argument("--frames_dir", type=str, default="/home/dgl/lxc/.runlogs/task3_frames")
    args = ap.parse_args()
    args.only_phases = {s.strip() for s in args.only_phases.split(",") if s.strip()}

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_task3_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=E.ENTRY_POINT,
        default_joint_values={},
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE

    # ---- 规范化初始场景（只动机器人关节到默认姿态，不动 free-joint 工具）----
    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()
    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    for c in manager.controllers:
        c.reset()
    env.render()
    time.sleep(0.05)

    # ---- 只读场景探测 ----
    print("\n[probe] === 当前 ORCA 场景探测 ===", flush=True)
    box = _site_pos(env, TOOLBOX_SITE)
    base = _body_pos(env, ROBOT_BASE_BODY)
    print(f"[probe] toolbox site {TOOLBOX_SITE}: {box}", flush=True)
    print(f"[probe] robot base  {ROBOT_BASE_BODY}: {base}", flush=True)
    reach_d = _dist(base, box)
    print(f"[probe] base→toolbox = {reach_d:.3f} m  (reach 满分带 ~0.4~1.0, 峰值 0.65)", flush=True)

    tool_init: dict = {}
    missing: list[str] = []
    for ph in SCORE_PHASES:
        p = _site_pos(env, ph["site"])
        if p is None:
            missing.append(ph["site"])
            print(f"[probe]  !! 缺少 site: {ph['site']}  step={ph['step_id']} prompt={ph['prompt']}", flush=True)
        else:
            tool_init[ph["site"]] = p.tolist()
            print(f"[probe]  {ph['prompt']:<10} dist_toolbox={_dist(p, box):6.3f}  @ {p}", flush=True)
    if missing:
        print(f"\n[probe] 缺少 {len(missing)} 个评分 site —— 当前布局不是 Task3 官方场景，无法评分！", flush=True)
        print("[probe] 请在 OrcaLab 加载含 Group_Interactive_*_task_*_site 的官方 Task3 布局。", flush=True)
        env.close(); return
    if base is None or box is None:
        print("\n[probe] 找不到机器人 base 或 toolbox site。", flush=True)
        env.close(); return
    if reach_d > 1.2 or reach_d < 0.3:
        print(f"[probe] 警告: base 距 toolbox {reach_d:.2f}m，reach 分可能偏低。", flush=True)

    # ---- 建立末端控制器与设备 ----
    l_arm = E.create_arm(env, E.agent_conf.l_arm)
    r_arm = E.create_arm(env, E.agent_conf.r_arm)
    kp = float(np.clip(args.kp, 1.0, 300.0))
    for _a in (l_arm, r_arm):
        _a.controller.kp = np.ones(6, dtype=np.float64) * kp
        _a.controller.kd = 2.0 * np.sqrt(_a.controller.kp)
    l_grip = E.create_gripper(env, E.agent_conf.gripper_l)
    r_grip = E.create_gripper(env, E.agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    _init_state = storage.build_state(storage.obs_callback(env))
    _init_apply = E.action_dict_for_apply(E.build_initial_action_from_state(_init_state))
    _init_apply["l_grip_ctrl"] = np.array([-0.8561, -0.8561], dtype=np.float32)
    device = E.EEFDevice(
        l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
        lock_left_arm=True, grasp_integral=args.grasp_integral, **_init_apply,
    )
    manager.set_device(device)
    device.set_left_hold(_init_apply["l_pos_b"], _init_apply["l_quat_b"], _init_apply.get("l_grip_ctrl"))
    for _ in range(8):
        act = manager.run_controllers()
        env.step(act)
        env.render()

    if args.dry_run:
        # 不连相机/策略，仅测 idle 控制步墙钟，供预算分配
        t0 = time.time(); N = 60
        for _ in range(N):
            act = manager.run_controllers()
            env.step(act)
            env.render()
        dt = (time.time() - t0) / N
        print(f"\n[dry-run] idle 控制步均值 {dt*1000:.1f} ms/step → 软时限 {SOFT_DEADLINE_S:.0f}s 约 {SOFT_DEADLINE_S/dt:.0f} 步",
              flush=True)
        print("[dry-run] 场景探测通过。工具初始位已记录。", flush=True)
        env.close()
        return

    # ---- 相机 + 策略 runner（评分路径才需要）----
    camera_map = E.omnipicker_camera_map(enable_wrist_l=False)
    camera_name_map: dict[str, str] = {e: k for e, (k, _port) in camera_map.items()}
    _need_cameras = not args.no_images
    _cameras: dict = {}
    _target_hw = E.DEFAULT_HW
    if _need_cameras:
        os.makedirs(E.STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(E.STREAM_TRIGGER_PATH)
        _cameras = E.bring_up_cameras(camera_map, port_timeout=30.0, frame_timeout=30.0)
        _target_hw = E.probe_camera_hw(_cameras, camera_map)
        print(f"[setup] 相机就绪 {len(_cameras)} 路 hw={_target_hw}", flush=True)
    policy_runner = E.OpenPIPolicyRunner(
        host=args.host, port=args.port, prompt="",
        camera_name_map=camera_name_map, cameras=_cameras,
        target_hw=_target_hw, use_images=not args.no_images,
    )
    print(f"[setup] 已连接策略服务 {args.host}:{args.port}", flush=True)
    if _need_cameras:
        E.warmup_camera_capture(manager, env, device, _init_apply, args.camera_warmup_steps)

    if args.smoke:
        print(f"\n[smoke] camera+policy 冒烟 N={args.smoke} 步（不评分）…", flush=True)
        policy_runner.prompt = SCORE_PHASES[2]["prompt"]  # 扳手（最近 0.36m）
        t0 = time.time()
        for _ in range(args.smoke):
            obs_state = storage.build_state(storage.obs_callback(env))
            chunk = policy_runner.infer_action_chunk(obs_state)
            for ma in chunk:
                pa = E.parse_policy_action(ma)
                device.set_target(**E.action_dict_for_apply(pa))
                act = manager.run_controllers()
                env.step(act)
                env.render()
        print(f"[smoke] 完成，均 {((time.time()-t0)/args.smoke)*1000:.1f} ms/步", flush=True)
        env.close()
        return

    # ---- 评分生命周期 ----
    scorer = None
    if not args.no_score:
        from orca_scorer_client import ScorerClient
        scorer = ScorerClient(team_id=args.team_id, team_token=args.team_token, robot_id=args.robot_id)
    if scorer:
        info = scorer.start_attempt("task3_tool_storage", prompt=f"{args.prompt}(Task3 工具收纳)")
        print(f"[scorer] attempt started: {info.get('attempt_id')}", flush=True)
    else:
        print("[scorer] 演练模式（无评分上报）", flush=True)

    _dump_frames(_cameras, "attempt_start", args.frames_dir)
    t_start = time.time()
    placed: dict[str, bool] = {}
    phase_logs: list[dict] = []
    attempt_started = bool(scorer)
    finished = False
    try:
        # ---- cycle 模式：逐件换指令 + 官方节奏(action_repeat10/kp220) ----
        # exp02/11999 每集只训练抓一把：放完一件后不会自动去抓下一件（只悬停）。
        # 因此每件入箱后立即把指令切到下一件未入箱工具；窗口超时则轮换，
        # 直到软时限或全部入箱。
        print(f"\n[run] cycle 模式 action_repeat={args.action_repeat} kp={kp} "
              f"每件窗口 {args.per_tool_max_s:.0f}s", flush=True)
        _dump_frames(_cameras, "run_start", args.frames_dir)

        step = 0
        last_log = time.time()
        box = _site_pos(env, TOOLBOX_SITE)
        placed = [False] * len(SCORE_PHASES)
        done_at: list[float | None] = [None] * len(SCORE_PHASES)
        for j, p in enumerate(SCORE_PHASES):  # 开场已在箱内的工具直接算完成
            c = _site_pos(env, p["site"])
            if c is not None and _dist(c, box) <= PLACED_DIST:
                placed[j] = True
                done_at[j] = 0.0

        from collections import deque
        queue = deque(j for j in range(len(SCORE_PHASES)) if not placed[j])
        target_win = time.time()
        while (time.time() - t_start) < SOFT_DEADLINE_S and step < args.max_steps and queue:
            ti = queue[0]
            if policy_runner.prompt != SCORE_PHASES[ti]["prompt"]:
                policy_runner.prompt = SCORE_PHASES[ti]["prompt"]
                target_win = time.time()
                print(f"[run] → 目标: {SCORE_PHASES[ti]['prompt']} (剩 {len(queue)} 件)", flush=True)
                _dump_frames(_cameras, f"target_{SCORE_PHASES[ti]['step_id']}", args.frames_dir)

            obs_state = storage.build_state(storage.obs_callback(env))
            chunk = policy_runner.infer_action_chunk(obs_state)
            for ma in chunk:
                if (time.time() - t_start) >= SOFT_DEADLINE_S or step >= args.max_steps or not queue:
                    break
                pa = E.parse_policy_action(ma)
                device.set_target(**E.action_dict_for_apply(pa))
                for _ in range(args.action_repeat):
                    act = manager.run_controllers()
                    _, _, _, truncated, _ = env.step(act)
                    env.render()
                    step += 1

                # 每 ~3s 遥测：当前目标距离 + 全部工具→box + EE + 夹爪
                if time.time() - last_log >= 3.0:
                    last_log = time.time()
                    box = _site_pos(env, TOOLBOX_SITE)
                    ee = _site_pos(env, EE_SITE_R)
                    ee_s = "n/a" if ee is None else f"({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f})"
                    dtgt = _dist(_site_pos(env, SCORE_PHASES[ti]["site"]), box)
                    ds = "  ".join(
                        f"{SCORE_PHASES[j]['prompt']}→{_dist(_site_pos(env, SCORE_PHASES[j]['site']), box):.3f}"
                        for j in range(len(SCORE_PHASES)))
                    print(f"    [t={time.time()-t_start:5.1f}s] 目标[{SCORE_PHASES[ti]['prompt']} {dtgt:.3f}] "
                          f"{ds}  ee={ee_s} grip={float(np.asarray(device.r_grip_ctrl)[0]):+.2f} step={step}",
                          flush=True)

                # 被动入箱检测
                box = _site_pos(env, TOOLBOX_SITE)
                changed = False
                for j in list(queue):
                    c = _site_pos(env, SCORE_PHASES[j]["site"])
                    if c is not None and box is not None and _dist(c, box) <= PLACED_DIST:
                        placed[j] = True
                        done_at[j] = time.time() - t_start
                        queue.remove(j)
                        changed = True
                        print(f"[run] ✓ {SCORE_PHASES[j]['prompt']} 已入箱 (t={done_at[j]:.1f}s)", flush=True)
                if changed:
                    _dump_frames(_cameras, "place_event", args.frames_dir)

                if placed[ti]:
                    break  # 当前目标已成 → 出 for，回 while 切换下一件

            # 当前目标窗口超时仍未入箱 → 轮换到队尾，给其它件机会
            if (time.time() - target_win) >= args.per_tool_max_s and queue and not placed[queue[0]]:
                f = queue.popleft()
                queue.append(f)
                policy_runner.prompt = ""  # 强制下轮重设 prompt
                print(f"[run] 窗口超时，{SCORE_PHASES[f]['prompt']} 未入箱 → 轮换", flush=True)
            if queue and placed[queue[0]]:
                queue.popleft()  # 兜底：队首已成则剔除

        el = time.time() - t_start
        for j, p in enumerate(SCORE_PHASES):
            phase_logs.append({"prompt": p["prompt"], "step_id": p["step_id"], "done": bool(placed[j]),
                               "elapsed_total": round(el, 1), "steps": step})
            mark = "已放入✓" if placed[j] else "未放入✗"
            tag = f" @ {done_at[j]:.1f}s" if done_at[j] else ""
            print(f"[run] {p['prompt']}: {mark}{tag}", flush=True)
        if all(placed):
            print("[run] 5 件全部入箱！", flush=True)
        elif (time.time() - t_start) >= SOFT_DEADLINE_S:
            print("[run] 软时限到，停止。", flush=True)
        else:
            print("[run] 步数上限/中断停止。", flush=True)

        if scorer and attempt_started and not finished:
            summary = scorer.finish()
            finished = True
            print(f"\n[scorer] score: {summary.get('score')} / {summary.get('max_score')}", flush=True)
            for r in summary.get("step_results", []):
                print(f"  {r.get('step_id')}: {'PASS' if r.get('success') else 'FAIL'}", flush=True)
            print("[scorer] 等待视频上传 25s …", flush=True)
            time.sleep(25)
    finally:
        if scorer and attempt_started and not finished:
            try:
                summary = scorer.finish()
                finished = True
                print(f"\n[scorer] (收尾) score: {summary.get('score')} / {summary.get('max_score')}", flush=True)
            except Exception as fe:
                print(f"[scorer] 收尾 finish 失败: {fe}", flush=True)
        if args.jsonl and phase_logs:
            Path(args.jsonl).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in phase_logs),
                                        encoding="utf-8")
        if _cameras:
            try: E.close_cameras(_cameras)
            except Exception: pass
        try: env.stop_save_video()
        except Exception: pass
        try: env.close()
        except Exception: pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[task3] 中断", flush=True)
    except Exception:
        import traceback; traceback.print_exc()
    finally:
        os._exit(0)
