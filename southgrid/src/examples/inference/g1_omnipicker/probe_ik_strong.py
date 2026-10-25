"""强 IK：用点云最近点当起点、**不加限位**反解训练帧位姿，判定真实可达性。

probe_reach5 的"覆盖度"只能分辨到 ~25mm（点云采样分辨率），且精修带 bounds，
目标又靠近工作空间边界 —— 边界处的解可能落在限位外被 bounds 挡住而假性失败。

这一版：
  · 起点 = 点云里离目标最近的 K 个位形（已经在正确盆地，不是随机撒点）
  · 反解**不加 bounds**，解出来再单独检查是否落在关节限位内
  · 位置先收敛，再在位置解附近补姿态
这样得到的结论才是"这个位姿在运动学上到底存不存在"。

用法：.../bin/python probe_ik_strong.py --n 20000 --starts 150
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
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=str, default="0,600,1800,2100")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--starts", type=int, default=150)
    args = ap.parse_args()

    import pandas as pd
    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)

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

    def fk7(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:14]

    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    print(f"[点云] {args.n} 个位形 …")
    rng = np.random.default_rng(5)
    Q = rng.uniform(lo, hi, size=(args.n, 7))
    t0 = time.time()
    P = np.array([fk7(q)[:3] for q in Q])
    print(f"[点云] {time.time()-t0:.1f}s")

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    def refine_pos(tgt_pos, q0):
        r = least_squares(lambda q: fk7(q)[:3] - tgt_pos, q0,
                          xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1500)
        return float(np.linalg.norm(fk7(r.x)[:3] - tgt_pos)), r.x.copy()

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    print(f"\n{'帧':>6s} {'目标位置':>30s} {'无约束最好':>11s} {'限位内?':>8s}  关节角")
    for fr in frames:
        if fr >= len(st):
            continue
        tgt7 = st[fr, 7:14]
        tgt = tgt7[:3]
        d = np.linalg.norm(P - tgt, axis=1)
        best = None
        t0 = time.time()
        for k in np.argsort(d)[:args.starts]:
            try:
                dp, q = refine_pos(tgt, Q[k])
            except Exception:
                continue
            if best is None or dp < best[0]:
                best = (dp, q)
        dp, q = best
        inlim = bool(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))
        print(f"{fr:>6d} {str(np.round(tgt,4).tolist()):>30s} {dp*1000:>8.2f} mm {str(inlim):>8s}  "
              f"{np.round(q,4).tolist()}  [{time.time()-t0:.0f}s]")
        if dp < 0.01:
            # 位置命中 → 在附近补姿态
            R_tgt = R.from_quat(tgt7[3:])
            r = least_squares(
                lambda qq: np.concatenate([
                    (fk7(qq)[:3] - tgt) * 100.0,
                    (R_tgt * R.from_quat(fk7(qq)[3:]).inv()).as_rotvec()]),
                q, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=3000)
            s2 = fk7(r.x)
            dp2 = float(np.linalg.norm(s2[:3] - tgt))
            da2 = float(np.degrees((R_tgt * R.from_quat(s2[3:]).inv()).magnitude()))
            inlim2 = bool(np.all(r.x >= lo - 1e-6) and np.all(r.x <= hi + 1e-6))
            print(f"       [补姿态] 位置 {dp2*1000:.3f} mm / 姿态 {da2:.3f}°  限位内={inlim2}")
            print(f"       关节角 = {np.round(r.x,4).tolist()}")
            print(f"       FK 校验 = {np.round(s2,4).tolist()}")
            print(f"       目标    = {np.round(tgt7,4).tolist()}")


if __name__ == "__main__":
    main()
