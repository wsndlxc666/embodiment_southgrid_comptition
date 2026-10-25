"""IK 反解采集时的起始关节角 —— 让右臂末端停在"训练帧 0"那个位姿。

背景（2026-09-17）：训练集 100 集的第一帧 state 完全相同，右臂末端停在
base 系 (+0.1585, -0.2792, +0.4356)，cycle_dist=0.510。而 ④ 用
`build_default_joint_values()`（右臂 = conf 的 neutral [1.42,-0.88,-1.54,1.48]）
teleport 之后读到的是 (+0.682,-0.080,+0.235)、cycle_dist=0.149 —— **差 52 cm，
且在训练集里找不到任何一帧能匹配**。也就是说模型从第一帧起就在分布外，
于是 0.5 秒就把爪子合了一下，把 cycle_idx 推过了第 0 轮。

同一个 `set_joint_qpos` 在采集器里也走，所以那组关节角跟现在 conf 里的
neutral 不是一回事。与其考据是哪次改动，这里直接用仿真当 FK 黑盒，
最小二乘反解出"能让末端停在训练帧 0 位姿"的 7 个关节角。

用法：
    .../bin/python probe_solve_start_pose.py
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

from scipy.spatial.transform import Rotation as R  # noqa: E402

# 训练帧 0 的右臂（base 系），来自 g1_omnipicker_clear5_d22_100ep（100 集全同）
TGT_POS = np.array([0.1585, -0.2792, 0.4356])
TGT_QUAT = np.array([-0.6987, 0.5574, 0.3993, 0.2043])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--quat_order", type=str, default="xyzw", choices=["xyzw", "wxyz"])
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=120)
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
        cycle_refs=cycle_refs,
    )
    storage.reset_cycle()
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")

    jn = list(E.agent_conf.r_arm["joint_names"])
    jidx = [env.joint(n) for n in jn]

    # 目标四元数：按需换序
    tq = TGT_QUAT.copy()
    if args.quat_order == "wxyz":
        tq = np.array([TGT_QUAT[3], TGT_QUAT[0], TGT_QUAT[1], TGT_QUAT[2]])
    R_tgt = R.from_quat(tq)

    n_eval = [0]
    t_first = [None]

    def fk(q: np.ndarray) -> np.ndarray:
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        n_eval[0] += 1
        if t_first[0] is None:
            t_first[0] = time.time()
        return s[7:14]

    # 用一次调用测单次耗时
    t0 = time.time()
    s0 = fk(np.zeros(7))
    dt = time.time() - t0
    print(f"[fk] 单次 set_joint_qpos+mj_forward+build_state 耗时 {dt*1000:.1f} ms"
          f"  （q=[0]*7 → 右臂 {np.round(s0[:3],4).tolist()}）")

    def residual(q: np.ndarray) -> np.ndarray:
        s = fk(q)
        rp, rq = s[:3], s[3:7]
        try:
            dR = R_tgt * R.from_quat(rq).inv()
            rv = dR.as_rotvec()
        except Exception:
            rv = np.full(3, 1.0)
        return np.concatenate([(rp - TGT_POS) * 10.0, rv])   # 位置加权 10 (米→分米量级)

    from scipy.optimize import least_squares
    rng = np.random.default_rng(0)
    starts = [np.zeros(7),
              np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64)] + \
             [rng.uniform(-1.5, 1.5, 7) for _ in range(max(0, args.restarts - 2))]
    best = None
    print(f"\n[ik] 目标：pos {np.round(TGT_POS,4).tolist()}  quat({args.quat_order}) {np.round(tq,4).tolist()}")
    for i, q0 in enumerate(starts):
        try:
            r = least_squares(residual, q0, max_nfev=args.maxiter * 8, xtol=1e-10, ftol=1e-10)
        except Exception as e:
            print(f"  start{i}: 求解异常 {e}")
            continue
        s = fk(r.x)
        dpos = float(np.linalg.norm(s[:3] - TGT_POS))
        try:
            dang = float(np.degrees((R_tgt * R.from_quat(s[3:7]).inv()).magnitude()))
        except Exception:
            dang = 999.0
        print(f"  start{i}: 位置误差 {dpos*1000:7.2f} mm   姿态误差 {dang:7.2f}°   q={np.round(r.x,4).tolist()}")
        if best is None or dpos < best[0]:
            best = (dpos, dang, r.x.copy(), s.copy())
    if best is None:
        print("求解失败")
        return
    dpos, dang, q, s = best
    print(f"\n=== 最佳解 ===")
    print(f"  关节角 {np.round(q,4).tolist()}")
    print(f"  位置误差 {dpos*1000:.2f} mm, 姿态误差 {dang:.2f}°")
    print(f"  解得末端 {np.round(s[:3],4).tolist()}, 目标 {np.round(TGT_POS,4).tolist()}")
    print(f"  cycle_dist 读得 {s[4]:.4f} (训练帧0 = 0.5100)")

    # 用这个解验证：teleport → 第一帧 22 维 state 与训练帧0 的逐维差
    env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
    env.mj_forward()
    storage.reset_cycle()
    full = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
    print(f"\n[验证] 用这组关节角 teleport 后，state[7:14] = {np.round(full[7:14],4).tolist()}")
    print(f"       训练帧0 state[7:14]              = {np.round(np.concatenate([TGT_POS, TGT_QUAT]),4).tolist()}")
    print(f"       右臂(7:14) 距离 = {np.linalg.norm(full[7:14]-np.concatenate([TGT_POS,TGT_QUAT])):.4f}")
    print(f"\n>>> 把这行加进 ④ 的初始化（teleport 之后、mj_forward 之前）：")
    print(f"    env.set_joint_qpos({{env.joint(n): np.float32(v) for n, v in zip(")
    print(f"        E.agent_conf.r_arm['joint_names'], {np.round(q,4).tolist()})}})")
    print(f"共 {n_eval[0]} 次 FK 求值，耗时 {time.time()-t_first[0]:.1f}s")


if __name__ == "__main__":
    main()
