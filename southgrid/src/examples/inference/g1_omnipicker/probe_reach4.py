"""判决性实验：拿训练集里**其他帧**（当年那台机器人确实到过的位姿）去反解。

逻辑：训练集第 0 帧的位姿当年一定可达（它被录下来了）。问题只是"今天的模型
还能不能摆出来"。所以不能只测帧 0 —— 得测一批**同一条轨迹上的其他帧**当基线：
  · 若今天的模型能命中它们、唯独命中不了帧 0  → 帧 0 是真不可达
  · 若连它们也大面积命中不了              → 今天这份模型和采集时不是同一套，
                                            那"用 IK 把起始位姿摆回去"这条路整个作废

管线已在 probe_reach3.py 标定过：对已知可达的旧数据集帧0，位置-only 反解
从 neutral 起就能到 0.062 mm，精确恢复 conf neutral。所以这里的读数可信。

用法：.../bin/python probe_reach4.py --frames 0,100,...,2000
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

LIMITS_DEG = [(-179.908752441, 179.908752441), (-119.748168945, 119.748168945),
              (-177.616912842, 177.616912842), (0.0, 100.267608643),
              (-177.616912842, 177.616912842), (-99.694656372, 99.694656372),
              (-177.616912842, 177.616912842)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--dataset", type=str, default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=str,
                    default="0,50,100,150,300,600,900,1100,1200,1500,1700,1800,1900,2000,2100")
    ap.add_argument("--starts", type=int, default=20)
    args = ap.parse_args()

    import pandas as pd
    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    df = pd.read_parquet(fs[args.episode])
    st = np.stack(df["observation.state"].values)
    print(f"[数据] {args.dataset} 第{args.episode}集 {st.shape}")

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

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    def fk_pos(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:10]

    # 顺带算：conf neutral 的 FK 位置，距训练位置云最近有多远
    neutral = np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64)
    p_neutral = fk_pos(neutral)
    d_all = np.linalg.norm(st[:, 7:10] - p_neutral, axis=1)
    k = int(d_all.argmin())
    print(f"\n[参考] conf neutral 的 FK 位置 = {np.round(p_neutral,4).tolist()}")
    print(f"       训练集里离它最近的帧 #{k}  pos={np.round(st[k,7:10],4).tolist()}"
          f"  纯位置距离 {d_all.min()*1000:.2f} mm"
          f"  （该帧右臂7维 {np.round(st[k,7:14],4).tolist()}）")

    from scipy.optimize import least_squares
    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])
    rng = np.random.default_rng(11)

    def best_pos(tgt, starts):
        b = None
        for k in range(starts):
            q0 = neutral if k == 0 else rng.uniform(lo, hi, 7)
            try:
                r = least_squares(lambda q: fk_pos(q) - tgt, q0, bounds=(lo, hi),
                                  xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=600)
            except Exception:
                continue
            d = float(np.linalg.norm(fk_pos(r.x) - tgt))
            if b is None or d < b[0]:
                b = (d, r.x.copy())
        return b

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    print(f"\n===== 逐帧位置-only 反解（每帧 {args.starts} 个起点）=====")
    print(f"{'帧':>6s} {'目标位置':>30s} {'最好位置误差':>14s}   结论")
    res = []
    t0 = time.time()
    for fr in frames:
        if fr >= len(st):
            continue
        tgt = st[fr, 7:10]
        d, q = best_pos(tgt, args.starts)
        res.append((fr, d))
        verdict = "✓ 可达" if d < 0.005 else ("△ 勉强" if d < 0.03 else "✗ 够不到")
        print(f"{fr:>6d} {str(np.round(tgt,4).tolist()):>30s} {d*1000:>11.2f} mm   {verdict}")
    print(f"\n[耗时 {time.time()-t0:.1f}s]")
    ok = [r for r in res if r[1] < 0.005]
    print(f"命中(<5mm): {len(ok)}/{len(res)} 帧")
    if len(ok) >= max(2, len(res) // 2):
        print("⇒ 今天的模型与采集时**一致**（多数帧都能摆回来）。帧0 若列在 ✗ 里，那是真的不可达。")
    else:
        print("⇒ 今天的模型与采集时**不一致**（连普通帧都摆不回来）。"
              "用 IK 把起始位姿摆回训练分布这条路作废。")
    for fr, d in res:
        if fr == 0:
            print(f"   [帧0 单列] 位置误差 {d*1000:.2f} mm")


if __name__ == "__main__":
    main()
