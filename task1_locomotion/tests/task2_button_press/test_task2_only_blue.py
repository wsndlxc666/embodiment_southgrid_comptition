"""Task2 按钮操作 - 仅蓝色按钮成功。

模拟机器人只正确识别并按压蓝色按钮，其余颜色均失败。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    step_map = {
        "press_red_button": False,
        "press_green_button": False,
        "press_blue_button": True,
        "press_yellow_button": False,
    }
    metrics = {
        "press_red_button": {"ee_to_button_m": 0.38},
        "press_green_button": {"ee_to_button_m": 0.41},
        "press_blue_button": {"ee_to_button_m": 0.07},
        "press_yellow_button": {"ee_to_button_m": 0.50},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_only_blue_store"),
        extra_metrics=metrics,
    )


if __name__ == "__main__":
    main()
