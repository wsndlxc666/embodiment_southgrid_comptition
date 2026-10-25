"""Task2 按钮操作 - 全失败测试用例。

模拟机器人完全未执行按压动作，所有步骤标记为未通过。
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
        "press_blue_button": False,
        "press_yellow_button": False,
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_all_fail_store"),
    )


if __name__ == "__main__":
    main()
