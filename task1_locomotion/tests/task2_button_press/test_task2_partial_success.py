"""Task2 按钮操作 - 部分成功测试用例。

机器人仅成功按压红、绿按钮，蓝、黄按钮未识别/未按压。
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
        "press_blue_button": False,
        "press_yellow_button": False,
    }
    metrics = {
        "press_red_button": {"ee_to_button_m": 0.08},
        "press_green_button": {"ee_to_button_m": 0.09},
        "press_blue_button": {"ee_to_button_m": 0.45},
        "press_yellow_button": {"ee_to_button_m": 0.52},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_partial_success_store"),
        extra_metrics=metrics,
    )


if __name__ == "__main__":
    main()
