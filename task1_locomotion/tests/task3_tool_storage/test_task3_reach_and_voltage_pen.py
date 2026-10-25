"""Task3 工具归纳 - 仅到达工具台并成功归位验电笔。

模拟机器人到达工具台后只完成一件工具的归位。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    step_map = {
        "reach_tool_table": True,
        "pick_voltage_pen": True,
        "pick_screwdriver": False,
        "pick_wrench": False,
        "pick_flashlight": False,
        "pick_electrician_knife": False,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_reach_and_voltage_pen_store"),
    )


if __name__ == "__main__":
    main()
