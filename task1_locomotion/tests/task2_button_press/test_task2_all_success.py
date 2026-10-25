"""Task2 按钮操作 - 全成功测试用例。

模拟机器人依次正确按压红、绿、蓝、黄四个按钮，所有步骤标记为通过。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    step_map = {
        "press_red_button": True,
        "press_green_button": True,
        "press_blue_button": True,
        "press_yellow_button": True,
    }
    metrics = {
        "press_red_button": {"ee_to_button_m": 0.08},
        "press_green_button": {"ee_to_button_m": 0.06},
        "press_blue_button": {"ee_to_button_m": 0.07},
        "press_yellow_button": {"ee_to_button_m": 0.05},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_all_success_store"),
        extra_metrics=metrics,
    )


if __name__ == "__main__":
    main()
