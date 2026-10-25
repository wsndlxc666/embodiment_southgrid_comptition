"""Task2 按钮操作 - 顺序/颜色错误测试用例。

模拟机器人把颜色识别错：按了蓝、黄按钮却以为按了红、绿。
所有步骤仍然被手动标记为通过，但额外信息记录实际按下的是错误按钮。
该用例用于验证评分客户端能正确上报并区分 step 语义。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from common_client import run_manual_attempt  # noqa: E402


def main() -> None:
    # 标记全部通过，但 extra 中写明实际动作与 step 语义不符
    step_map = {
        "press_red_button": True,
        "press_green_button": True,
        "press_blue_button": True,
        "press_yellow_button": True,
    }
    extra = {
        "press_red_button": {"actual_pressed": "blue", "match": False},
        "press_green_button": {"actual_pressed": "yellow", "match": False},
        "press_blue_button": {"actual_pressed": "red", "match": False},
        "press_yellow_button": {"actual_pressed": "green", "match": False},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_wrong_buttons_store"),
        extra_metrics=extra,
    )


if __name__ == "__main__":
    main()
