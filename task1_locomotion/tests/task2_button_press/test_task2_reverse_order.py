"""Task2 按钮操作 - 倒序全部成功。

验证步骤提交顺序不影响最终计分：按黄、蓝、绿、红倒序标记全部通过。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    #  intentionally reverse the natural order
    step_map = {
        "press_yellow_button": True,
        "press_blue_button": True,
        "press_green_button": True,
        "press_red_button": True,
    }
    metrics = {
        "press_yellow_button": {"ee_to_button_m": 0.06},
        "press_blue_button": {"ee_to_button_m": 0.05},
        "press_green_button": {"ee_to_button_m": 0.07},
        "press_red_button": {"ee_to_button_m": 0.08},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_reverse_order_store"),
        extra_metrics=metrics,
    )


if __name__ == "__main__":
    main()
