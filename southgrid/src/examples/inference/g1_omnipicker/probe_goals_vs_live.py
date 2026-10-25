"""`goals` 表里的点，和**场景里实际的**工具/盒子，是不是同一个地方？

为什么要问：runner 发给模型的 state 第 18 维 `goal_dist` 取的是静态表
`tool_refs_d22.json::goals`，而日志里"末端离各工具"取的是现场 query 的 site。
两者对不上：对齐到训练帧0（残差 3.16mm）那一刻，
    goal_dist(静态表, 螺丝刀) = 0.4784   ← 与训练帧0 的 0.4784 一致
    现场 |EE − 螺丝刀 site|    = 0.580    ← 差 102mm
若静态表来自**采集时**的场景，那就说明**评测场景里工具没摆在采集时的位置上**，
模型收到的图像/几何与训练分布不一致 —— 这正是"视觉不起作用"的候选解释。
若不来自采集场景，那就只是表本身没校准。

本脚本把三样东西并排，全在**机器人 base 系**下：
  A 静态表 goals[0..4]（5 个工具抓取点）与 goals[5]（盒子落点）
  B 现场 query 的 5 个工具 site + 盒子 site（世界系 → 转 base 系）
  C 逐个算 |A−B|，看差多少

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_goals_vs_live.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402
from run_task3_tool_storage_scored import (  # noqa: E402
    ROBOT_BASE_BODY, SCORE_PHASES, TOOLBOX_SITE)

TR = os.path.join(base_dir, "tool_refs_d22.json")


def main() -> None:
    from yaml import Loader, load

    cfg = os.path.abspath(os.path.join(base_dir, "../../dataCollection/common/example.yaml"))
    with open(cfg, "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    sm = E.SceneManager("localhost:50051", config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=sm, frame_skip=5, orcagym_addr="localhost:50051")
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset()
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()

    with open(TR, "r", encoding="utf-8") as f:
        tr = json.load(f)
    goals = np.asarray(tr["goals"], dtype=np.float64)

    sites = [p["site"] for p in SCORE_PHASES] + [TOOLBOX_SITE]
    names = [p["prompt"] for p in SCORE_PHASES] + ["盒子"]

    # 现场 site 世界坐标 → base 系
    got = env.query_site_pos_and_quat_B(sites, [ROBOT_BASE_BODY])
    live = np.asarray([np.asarray(got[s]["xpos"], dtype=np.float64).reshape(3)
                       for s in sites], dtype=np.float64)

    print("静态表 goals 的坐标系取自 runner 自己的口径（与 state 第19维 goal_dist 同源）")
    print(f"{'':10s} {'静态表 goals':>34s}   {'现场 site (base系)':>34s}   {'差':>8s}")
    for i, nm in enumerate(names):
        g = goals[i] if i < len(goals) else None
        if g is None:
            continue
        d = float(np.linalg.norm(g - live[i]))
        print(f"{nm:10s} {np.round(g, 4).tolist()!s:>34s}   "
              f"{np.round(live[i], 4).tolist()!s:>34s}   {d*1000:6.1f}mm")
    print(f"\n（goals 共 {len(goals)} 个点；现场只比了前 {len(names)} 个）")

    # 现场工具**互相之间的**相对位置，和静态表的相对位置比 —— 这一项与 base 系原点无关，
    # 若也差，就是场景本身摆得不一样（而不是 base 位姿差）。
    print("\n相对几何（把各点减去第 0 个工具，消掉 base 位姿的影响）：")
    print(f"{'':10s} {'静态表(相对)':>34s}   {'现场(相对)':>34s}   {'差':>8s}")
    for i, nm in enumerate(names):
        if i >= len(goals):
            continue
        dg = goals[i] - goals[0]
        dl = live[i] - live[0]
        print(f"{nm:10s} {np.round(dg, 4).tolist()!s:>34s}   "
              f"{np.round(dl, 4).tolist()!s:>34s}   {np.linalg.norm(dg-dl)*1000:6.1f}mm")

    # 机器人 base 在世界哪儿，留档（换过一次 base 位姿就是这里能看出来）
    try:
        b = env.query_site_pos_and_quat([TOOLBOX_SITE])
        print("\n盒子 site 世界坐标 =",
              np.round(np.asarray(b[TOOLBOX_SITE]["xpos"], dtype=np.float64), 5).tolist())
    except Exception as exc:
        print(f"\n盒子 site 世界坐标读取失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
