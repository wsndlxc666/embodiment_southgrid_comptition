"""定姿：利用 j5/j7 不影响末端位置这一结构，在位置解上把姿态也解出来。

现状（2026-09-17）：
  · 训练帧 0 的位置用无约束 IK 能到 2.91 mm 且在限位内 —— 位置可达。
  · 但同一次求解姿态差 134.7°。帧 0 是点云覆盖最差的位姿(59.5mm)，属工作空间
    **内边界**（肘关节限位 100° 卡住了收拢），位置约束把 j1~j4,j6 基本钉死，
    只剩 j5/j7 两个自由度去凑 3 个姿态自由度 —— 所以姿态差得多。
  · 已验证：旧数据集帧 0 的 pos 与 quat 都能被今天的 FK 复现到 2e-4，
    所以模型和四元数约定都是对的，不是"模型变了"。

这一版：先把位置解出来（点云最近点作起点），再在位置解附近
  ① 对 j5/j7 做网格定姿（它们不动位置，是免费的姿态自由度）
  ② 再用加权残差 [pos*100, rotvec] 整体抛光
最后报告 pos/orient 误差与是否在限位内。

用法：.../bin/python probe_ik_final.py --n 12000 --starts 120
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
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--grid", type=int, default=48, help="j5/j7 各自网格点数")
    args = ap.parse_args()

    import pandas as pd
    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt7 = st[args.frame, 7:14].astype(np.float64)
    print(f"[目标] 第{args.episode}集 帧{args.frame} 右臂 = {np.round(tgt7,4).tolist()}")

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

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    R_tgt = R.from_quat(tgt7[3:])

    def ori_err(s):
        try:
            return float(np.degrees((R_tgt * R.from_quat(s[3:]).inv()).magnitude()))
        except Exception:
            return 999.0

    # ---- 1. 位置解 ----
    print(f"[点云] {args.n} …")
    rng = np.random.default_rng(9)
    Q = rng.uniform(lo, hi, size=(args.n, 7))
    t0 = time.time()
    P = np.array([fk7(q)[:3] for q in Q])
    print(f"[点云] {time.time()-t0:.1f}s")
    d = np.linalg.norm(P - tgt7[:3], axis=1)

    print(f"[位置] 从最近 {args.starts} 个点云位形精修 …")
    cands = []
    for k in np.argsort(d)[:args.starts]:
        try:
            r = least_squares(lambda q: fk7(q)[:3] - tgt7[:3], Q[k],
                              xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1200)
        except Exception:
            continue
        s = fk7(r.x)
        cands.append((float(np.linalg.norm(s[:3] - tgt7[:3])), r.x.copy()))
    cands.sort(key=lambda x: x[0])
    print(f"  最好位置误差 {cands[0][0]*1000:.3f} mm；取前 {min(8,len(cands))} 个做定姿")

    # ---- 2. j5/j7 网格定姿 ----
    best = None
    for dp, q0 in cands[:8]:
        g5 = np.linspace(max(lo[4], q0[4] - 1.2), min(hi[4], q0[4] + 1.2), args.grid)
        g7 = np.linspace(max(lo[6], q0[6] - 1.2), min(hi[6], q0[6] + 1.2), args.grid)
        for a in g5:
            for b in g7:
                q = q0.copy(); q[4] = a; q[6] = b
                s = fk7(q)
                dp2 = float(np.linalg.norm(s[:3] - tgt7[:3]))
                if dp2 > 0.02:
                    continue
                da = ori_err(s)
                score = dp2 * 100.0 + np.radians(da) * 0.05
                if best is None or score < best[0]:
                    best = (score, dp2, da, q.copy())
    if best is None:
        print("网格定姿全部位置超差，退化为只用位置解")
        _, dp, da, q = 0.0, cands[0][0], ori_err(fk7(cands[0][1])), cands[0][1]
        best = (0.0, dp, da, q)
    _, dp, da, q = best
    print(f"  [网格后] 位置 {dp*1000:.3f} mm / 姿态 {da:.3f}°")

    # ---- 3. 整体抛光 ----
    r = least_squares(
        lambda qq: np.concatenate([(fk7(qq)[:3] - tgt7[:3]) * 100.0,
                                   (R_tgt * R.from_quat(fk7(qq)[3:]).inv()).as_rotvec()]),
        q, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=4000)
    q, s = r.x, fk7(r.x)
    dp = float(np.linalg.norm(s[:3] - tgt7[:3])); da = ori_err(s)
    inlim = bool(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))

    print(f"\n===== 最终 =====")
    print(f"  关节角   = {np.round(q,4).tolist()}")
    print(f"  FK       = {np.round(s,4).tolist()}")
    print(f"  目标     = {np.round(tgt7,4).tolist()}")
    print(f"  位置误差 = {dp*1000:.3f} mm   姿态误差 = {da:.3f}°   限位内 = {inlim}")
    if dp < 0.005 and da < 5.0 and inlim:
        print(f"\n>>> 可用。把它加进 ④ 的初始化（teleport 之后、mj_forward 之前）：")
        print(f"    env.set_joint_qpos({{env.joint(n): np.float32(v) for n, v in zip(")
        print(f"        E.agent_conf.r_arm['joint_names'], {np.round(q,4).tolist()})}})")


if __name__ == "__main__":
    main()
