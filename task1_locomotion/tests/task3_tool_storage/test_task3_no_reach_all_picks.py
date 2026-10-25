"""Task3 工具归纳 - 未到达工具台但所有拾取归位成功。

验证各步骤独立计分：即使 reach_tool_table 失败，后续 pick 步骤仍应得分。
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
        "pick_voltage_pen": True,
        "pick_screwdriver": True,
        "pick_wrench": True,
        "pick_flashlight": True,
        "pick_electrician_knife": True,
    }
    run_manual_attempt(
        task_id="task3_tool_storage",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task3_no_reach_all_picks_store"),
    )


if __name__ == "__main__":
    main()
