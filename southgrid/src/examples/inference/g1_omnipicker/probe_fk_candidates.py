"""廉价穷举：clear5 帧 0 的右臂位姿，是不是某个"符号/取值变体"的 neutral。

推理：采集器和 ④ 走同一行代码 `set_default_joint_values(conf.r_arm.neutral)`，
今天这组值给 (0.6822,-0.0802,0.2355)，而 clear5 帧 0 是 (0.1585,-0.2792,0.4356)。
所以采集时那 7 个数不一样。与其反复做 IK，不如**把可能的变体直接 FK 一遍**：
符号翻转、取左臂的 neutral、镜像、前 4 关节的各种符号组合……

每次 FK 只要 3ms，几百个候选也就一两秒。

用法：.../bin/python probe_fk_candidates.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

TGT = np.array([0.1585, -0.2792, 0.4356, -0.6987, 0.5574, 0.3993, 0.2043])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    args = ap.parse_args()

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)
    with open(args.tool_refs, "r", encoding="utf-8") as f:
        _tr = json.load(f)
    tool_refs = np.asarray(_tr.get("goals", _tr["tool_refs"]), dtype=np.float32).reshape(-1, 3)
    ref_name = "goal_dist" if "goals" in _tr else "tool_dist"
    cycle_refs = (np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
                  if _tr.get("new_dims") else None)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(
        dataset_path="/tmp/_probe_scratch", tool_refs=tool_refs, ref_name=ref_name,
        cycle_refs=cycle_refs)
    storage.reset_cycle()
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr=args.orcagym_addr)
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    def fk7(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:14]

    R_neutral = list(E.agent_conf.r_arm["neutral_joint_values"])
    R_left = list(E.agent_conf.l_arm["neutral_joint_values"])

    cands: list[tuple[str, list]] = []
    cands.append(("今天的 conf.r_arm.neutral", R_neutral))
    cands.append(("conf.l_arm.neutral 原样", R_left))
    cands.append(("全零", [0.0] * 7))
    cands.append(("r_arm.neutral 全取负", [-v for v in R_neutral]))
    cands.append(("l_arm.neutral 全取负", [-v for v in R_left]))
    # 前 4 个关节的所有符号组合（后 3 个为 0）
    base = R_neutral
    for bits in itertools.product([1, -1], repeat=4):
        q = [base[i] * bits[i] for i in range(4)] + [0.0, 0.0, 0.0]
        cands.append(("前4关节符号 " + "".join("+" if b > 0 else "-" for b in bits), q))
    # 前 4 个关节符号组合 + 后 3 关节自由（每个 ±1.42 / ±0.88 之类的小集合）
    for bits in itertools.product([1, -1], repeat=4):
        for j567 in ([0, 0, 0], [1.42, 0.88, 0.0], [-1.42, -0.88, 0.0]):
            q = [base[i] * bits[i] for i in range(4)] + list(j567)
            cands.append(("前4" + "".join("+" if b > 0 else "-" for b in bits) + f" 后3={j567}", q))

    print(f"[穷举] {len(cands)} 个候选")
    rows = []
    for name, q in cands:
        s = fk7(np.asarray(q, dtype=np.float64))
        dp = float(np.linalg.norm(s[:3] - TGT[:3]))
        rows.append((dp, name, q, s))
    rows.sort(key=lambda x: x[0])

    print(f"\n目标 pos {np.round(TGT[:3],4).tolist()}  quat {np.round(TGT[3:],4).tolist()}")
    print(f"{'位置误差':>10s}  {'候选':<34s} {'FK pos':>28s}")
    for dp, name, q, s in rows[:12]:
        print(f"{dp*1000:>7.1f} mm  {name:<34s} {str(np.round(s[:3],4).tolist()):>28s}")

    dp, name, q, s = rows[0]
    print(f"\n最好候选: {name}")
    print(f"  q  = {q}")
    print(f"  FK = {np.round(s,4).tolist()}")
    if dp < 0.02:
        print("  ⇒ 命中！这个变体就是采集时用的那一组。")
    else:
        print(f"  ⇒ 都不是。最接近的也差 {dp*1000:.0f} mm，说明采集时那组值不在这些常见变体里。")


if __name__ == "__main__":
    main()
