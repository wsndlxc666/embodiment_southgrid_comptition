"""Task3 工具归纳收置 - 仅到达测试用例。

机器人只完成了“到达作业位姿”，所有工具拾取/归位步骤均未执行。
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
        "pick_voltage_pen": False,
        "pick_screwdriver": False,
        "pick_wrench": False,
        "pick_flashlight": False,
        "pick_electrician_knife": False,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_reach_only_store"),
    )


if __name__ == "__main__":
    main()
