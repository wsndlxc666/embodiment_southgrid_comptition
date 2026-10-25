"""判决：clear5 的 state 里 dims 0-6 / 7-13 到底哪个是左臂？

起因：clear5 帧 0 的 dims 7-13 = (0.1585,-0.2792,0.4356)，而今天 conf 的右臂 neutral
FK 出来是 (0.6822,-0.0802,0.2354) —— 差 127mm；我把 69 种符号/取值变体全 FK 了一遍，
最接近的也差 **412mm**。说明"采集时那 7 个关节角是 neutral 的某个变体"这个假设是错的。

另一个假设便宜得多：**当时左臂固定在 _L_INIT_JOINT_VALUES = [0]*7**（采集器第 89 行），
而两臂在 state 里的槽位如果和现在相反，那 dims 7-13 记的就是**左臂**的末端位姿 ——
一个全零关节的位姿，跟右臂 neutral 的任何变体都差得远，正好解释那 412mm。

这一版直接把左右臂摆在几组已知关节角上 FK，把两臂的末端位姿都打出来，
再和两个数据集帧 0 的 dims 0-6 / 7-13 逐一对号。

用法：.../bin/python probe_arm_order.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--clear5", type=str, default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--old", type=str, default="/home/dgl/lxc/openpi/data/g1_omnipicker_tool")
    args = ap.parse_args()

    import pandas as pd

    def frame0(ds, n=14):
        fs = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
        df = pd.read_parquet(fs[0])
        st = np.stack(df["observation.state"].values)
        return st, st[0][:n]

    print("=" * 78)
    for tag, ds in (("clear5_d22_100ep", args.clear5), ("g1_omnipicker_tool(旧)", args.old)):
        try:
            st, s0 = frame0(ds)
        except Exception as e:
            print(f"[{tag}] 读不到: {e}")
            continue
        print(f"[{tag}] {len(st)} 帧  state 维数 = {st.shape[1]}")
        print(f"  帧0 dims 0:7  = {np.round(s0[0:7], 4).tolist()}")
        print(f"  帧0 dims 7:14 = {np.round(s0[7:14], 4).tolist()}")
        if st.shape[1] > 14:
            print(f"  帧0 14:22     = {np.round(s0[14:22], 4).tolist()}")
    print("=" * 78)

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

    lj = [env.joint(n) for n in E.agent_conf.l_arm["joint_names"]]
    rj = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    def fk(lq, rq):
        d = {}
        for j, v in zip(lj, lq):
            d[j] = np.float32(v)
        for j, v in zip(rj, rq):
            d[j] = np.float32(v)
        env.set_joint_qpos(d)
        env.mj_forward()
        storage.reset_cycle()
        return np.asarray(storage.build_state(storage.obs_callback(env)),
                          dtype=np.float64).reshape(-1)

    L_zero = [0.0] * 7
    R_zero = [0.0] * 7
    L_n = list(E.agent_conf.l_arm["neutral_joint_values"])
    R_n = list(E.agent_conf.r_arm["neutral_joint_values"])

    print("FK 对照（dims 0:7 与 7:14 都打出来，看哪个槽位对上）")
    print(f"{'左臂关节':<26s} {'右臂关节':<26s} {'FK dims 0:7':>34s} {'FK dims 7:14':>34s}")
    cases = [
        ("全零", L_zero, "全零", R_zero),
        ("全零", L_zero, "conf neutral", R_n),
        ("conf neutral", L_n, "全零", R_zero),
        ("conf neutral", L_n, "conf neutral", R_n),
    ]
    for ln, lq, rn, rq in cases:
        s = fk(lq, rq)
        print(f"{ln:<26s} {rn:<26s} {str(np.round(s[0:7], 4).tolist()):>34s} "
              f"{str(np.round(s[7:14], 4).tolist()):>34s}")

    print("\n判读：把上面任一行和数据集帧 0 的两行对号。"
          "\n  · 若 clear5 帧0 的 7:14 等于「左臂全零」的 **dims 0:7** → 采集时两臂槽位是反的。"
          "\n  · 若等于任何一行的 7:14 → 槽位没反，就是采集时那组关节角和今天 conf 不同。")


if __name__ == "__main__":
    main()
