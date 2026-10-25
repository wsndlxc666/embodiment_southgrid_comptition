"""OrcaLocomotion 评分客户端测试公共模块。

此模块依赖通过 pip 安装的 competition_scoring_client 包中的 CompetitionClient，
不再通过 sys.path 引用外部源码目录。
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Optional

from orca_competition.client import CompetitionClient
from orca_scorer_client import _get_config

# 默认评分服务器与队伍凭证（用户指定 team_004）
# 支持优先级：环境变量 > orca_scoring.yaml 配置文件 > 内置默认值
DEFAULT_SERVER_URL = _get_config(
    "server_url", "ORCA_SCORING_SERVER_URL", "http://8.159.130.194/"
)
DEFAULT_TEAM_ID = _get_config("team_id", "ORCA_SCORING_TEAM_ID", "team_004")
DEFAULT_TEAM_TOKEN = _get_config("team_token", "ORCA_SCORING_TEAM_TOKEN", "token_team_004")

TASK2_STEPS = [
    ("press_red_button", 10.0),
    ("press_green_button", 10.0),
    ("press_blue_button", 10.0),
    ("press_yellow_button", 10.0),
]

TASK3_STEPS = [
    ("reach_tool_table", 5.0),
    ("pick_voltage_pen", 7.0),
    ("pick_screwdriver", 7.0),
    ("pick_wrench", 7.0),
    ("pick_flashlight", 7.0),
    ("pick_electrician_knife", 7.0),
]


def run_manual_attempt(
    task_id: str,
    step_success_map: dict[str, bool],
    server_url: str = DEFAULT_SERVER_URL,
    team_id: str = DEFAULT_TEAM_ID,
    team_token: str = DEFAULT_TEAM_TOKEN,
    local_store: Optional[str] = None,
    extra_metrics: Optional[dict[str, dict[str, Any]]] = None,
    record_video: bool = True,
    orca_addr: str = "localhost:50051",
    video_dir: str = "./recordings",
) -> dict[str, Any]:
    """手动驱动一次任务尝试，按 step_success_map 设置各步骤成败。

    Args:
        task_id: 任务 ID，如 task2_button_press / task3_tool_storage。
        step_success_map: {step_id: bool}，决定每个步骤是否通过。
        server_url: 中央评分服务器地址。
        team_id / team_token: 队伍凭证。
        local_store: 本地存储目录，默认在脚本目录下 .orca_comp/<task_id>。
        extra_metrics: 可选的 {step_id: {key: value}} 过程指标。
        record_video: 是否启用 Windows 屏幕录制（需 ORCA 窗口存在）。
        orca_addr: ORCA gRPC 地址，用于视频录制连接。
        video_dir: 视频本地保存目录。

    Returns:
        尝试汇总字典，包含 score / max_score / step_results 等。
    """
    if local_store is None:
        local_store = os.path.join(
            ".orca_comp_locomotion", f"{team_id}_{task_id}_store"
        )

    video_config: Optional[dict[str, Any]] = None
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        video_config = {
            "enabled": True,
            "orca_addr": orca_addr,
            "capture_mode": "SYNC",
            "local_dir": video_dir,
            "auto_upload": False,
            "use_screen_capture": True,
            "use_orca_builtin": False,
            "use_local_fallback": True,
            "fps": 20,
            "resolution": (1280, 720),
            "window_keyword": "CSG_RobotCompetition_2026",
            "window_width": 1280,
            "window_height": 720,
            "window_restore": True,
            "capture_scale": 1.0,
        }

    client = CompetitionClient(
        team_id=team_id,
        team_token=team_token,
        server_url=server_url,
        local_store=local_store,
        video_config=video_config,
    )

    try:
        attempt = client.start_task(task_id=task_id)
        print(f"[attempt] {attempt.attempt_id} task={task_id} started")

        for step_id, _score in step_success_map.items():
            step_start = time.time()
            with attempt.step(step_id) as step:
                ok = bool(step_success_map.get(step_id, False))
                step.set_success(ok)
                step.set_extra("test_type", "manual")
                if extra_metrics and step_id in extra_metrics:
                    for k, v in extra_metrics[step_id].items():
                        step.record_metric(k, v)
                # 手动用例虽无真实仿真时间，但模拟一个较小的执行耗时并把 wall-clock
                # 时长作为 duration/s sim_time 上报，供服务端计算平均耗时等指标。
                time.sleep(random.uniform(0.15, 0.35))
                elapsed = time.time() - step_start
                step.record_metric("duration_s", round(elapsed, 4))
                step.record_metric("sim_time_s", round(elapsed, 4))
                step.set_sim_time(elapsed)
                print(f"  {step_id}: {'PASS' if ok else 'FAIL'}  duration={elapsed:.3f}s")

        summary = client.end_task(attempt, status="success")

        # StepResult 本身不带 score，需用本地任务步骤分值表计算
        score_map = dict(TASK2_STEPS) if task_id == "task2_button_press" else dict(TASK3_STEPS)
        result = {
            "attempt_id": summary.attempt_id,
            "task_id": summary.task_id,
            "score": sum(
                score_map.get(r.step_id, 0.0) for r in summary.step_results if r.success
            ),
            "max_score": sum(score_map.values()),
            "success_steps": summary.success_steps,
            "total_steps": summary.total_steps,
            "step_results": [
                {
                    "step_id": r.step_id,
                    "success": r.success,
                    "score": score_map.get(r.step_id, 0.0),
                    "duration_s": r.duration_s,
                }
                for r in summary.step_results
            ],
        }
        print(
            f"[summary] score={result['score']}/{result['max_score']} "
            f"steps={result['success_steps']}/{result['total_steps']}"
        )
        return result
    finally:
        client.close()
