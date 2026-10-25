"""Task3 工具归纳收置 - 部分成功测试用例。

机器人成功到达工具台，但只归位了验电笔和螺丝刀，其余工具掉落/未放置成功。
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
        "pick_wrench": False,
        "pick_flashlight": False,
        "pick_electrician_knife": False,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_partial_success_store"),
    )


if __name__ == "__main__":
    main()
