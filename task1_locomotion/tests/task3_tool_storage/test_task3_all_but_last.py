"""Task3 工具归纳 - 前 5 步成功，最后一件电工刀失败。

模拟机器人几乎完成所有工具归位，但在最后一件工具处失败。
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
        "pick_screwdriver": True,
        "pick_wrench": True,
        "pick_flashlight": True,
        "pick_electrician_knife": False,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_all_but_last_store"),
    )


if __name__ == "__main__":
    main()
