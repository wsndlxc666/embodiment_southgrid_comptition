"""Task2 按钮操作 - 全部标记成功但末端距离贴近阈值。

模拟按压动作接近成功边界，记录 ee_to_button_m 指标，验证服务端接收指标。
实际 success 仍由手动标记决定，本用例侧重指标上报与汇总。
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
        "press_blue_button": True,
        "press_yellow_button": True,
    }
    # 距离均在 0.10m 阈值附近或略超，但手动标记为成功
    metrics = {
        "press_red_button": {"ee_to_button_m": 0.095},
        "press_green_button": {"ee_to_button_m": 0.100},
        "press_blue_button": {"ee_to_button_m": 0.105},
        "press_yellow_button": {"ee_to_button_m": 0.098},
    }
    run_manual_attempt(
        task_id="task2_button_press",
        step_success_map=step_map,
        local_store=os.path.join(HERE, ".task2_near_miss_store"),
        extra_metrics=metrics,
    )


if __name__ == "__main__":
    main()
