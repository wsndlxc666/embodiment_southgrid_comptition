"""Task3 工具归纳收置 - 全失败测试用例。

模拟机器人既未到达工具台，也未完成任何工具归位。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    step_map = {
        "reach_tool_table": False,
        "pick_voltage_pen": False,
        "pick_screwdriver": False,
        "pick_wrench": False,
        "pick_flashlight": False,
        "pick_electrician_knife": False,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_all_fail_store"),
    )


if __name__ == "__main__":
    main()
