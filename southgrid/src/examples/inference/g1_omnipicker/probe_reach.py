"""可达性扫描 —— clear5 训练帧 0 的右臂位姿今天到底能不能摆出来。

背景（2026-09-17 排查）：四份数据集第 0 帧对比
    旧的 tool / single_grasp / single_place : 右臂 (0.6822, -0.0802, 0.2354)
    clear5 / clear5_d22_100ep              : 右臂 (0.1585, -0.2792, 0.4356)
而 `g1_omnipicker_conf.py` 今天这版 neutral [1.42,-0.88,-1.54,1.48,0,0,0] 的 FK
**正好等于旧数据集的值**。所以 clear5 采集时右臂的初始关节角是另一组，
今天这份 conf 复现不出来。

`probe_solve_start_pose.py` 用 6 个起点的 least_squares 只到 370 mm，
那可能只是局部极小。这里做全局随机扫描：在 XML 关节限位内均匀撒点，
看"离目标最近能到多少"。同时拿旧数据集的帧 0 当**对照组** ——
如果连它都命中不了，说明是扫描本身写错了，而不是目标不可达。

用法：
    .../bin/python probe_reach.py --n 6000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

# 目标：训练帧 0 的右臂 7 维 = pos(3) + quat(3+1)，base 系
TGT_CLEAR5 = np.array([0.1585, -0.2792, 0.4356, -0.6987, 0.5574, 0.3993, 0.2043])
# 对照组：旧数据集帧 0 的右臂 7 维（= 今天 conf neutral 的 FK）
TGT_OLD = np.array([0.6822, -0.0802, 0.2354, -0.6258, -0.2309, 0.3103, 0.6773])

# XML 关节限位（弧度），从场景 XML 的 range 字段换算；顺序 = idx61..idx67
LIMITS_DEG = [(-179.908752441, 179.908752441),
              (-119.748168945, 119.748168945),
              (-177.616912842, 177.616912842),
              (0.0, 100.267608643),
              (-177.616912842, 177.616912842),
              (-99.694656372, 99.694656372),
              (-177.616912842, 177.616912842)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--n", type=int, default=6000, help="随机采样数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine", type=int, default=3, help="对最好的几个解做 least_squares 精修")
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

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    def fk7(q: np.ndarray) -> np.ndarray:
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:14]

    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    # 单次耗时 + 几何自检
    t0 = time.time()
    s_neutral = fk7(np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64))
    dt = time.time() - t0
    print(f"[fk] 单次 {dt*1000:.1f} ms")
    print(f"[几何] conf neutral 的 FK   = {np.round(s_neutral, 4).tolist()}")
    print(f"[几何] 旧数据集帧0(对照组)  = {np.round(TGT_OLD, 4).tolist()}"
          f"   距离 {np.linalg.norm(s_neutral - TGT_OLD):.4f}")
    print(f"[几何] 限位 (deg) = " + ", ".join(f"[{a:.0f},{b:.0f}]" for a, b in LIMITS_DEG))

    rng = np.random.default_rng(args.seed)
    n = args.n
    Q = rng.uniform(lo, hi, size=(n, 7))
    S = np.empty((n, 7))
    t0 = time.time()
    for i in range(n):
        S[i] = fk7(Q[i])
        if (i + 1) % 1000 == 0:
            print(f"    扫描 {i+1}/{n} … {time.time()-t0:.1f}s")
    print(f"[扫描] {n} 个随机位形，耗时 {time.time()-t0:.1f}s")

    for name, tgt in (("clear5 帧0", TGT_CLEAR5), ("旧数据集帧0", TGT_OLD)):
        dp = np.linalg.norm(S[:, :3] - tgt[:3], axis=1)          # 只看位置
        d7 = np.linalg.norm(S - tgt, axis=1)                     # 位置+四元数
        i_p, i_7 = int(dp.argmin()), int(d7.argmin())
        print(f"\n=== {name} ===")
        print(f"  纯位置最近: {dp[i_p]:.4f} m   q={np.round(Q[i_p], 4).tolist()}")
        print(f"              得 {np.round(S[i_p, :3], 4).tolist()}  目标 {np.round(tgt[:3], 4).tolist()}")
        print(f"  7维最近   : {d7[i_7]:.4f}      q={np.round(Q[i_7], 4).tolist()}")
        if dp[i_p] > 0.05:
            print(f"  ⇒ 纯位置都够不到（>5cm），该位姿在今天的模型下**不可达**")
        else:
            print(f"  ⇒ 位置可达，差在姿态")

    # 精修 clear5 目标
    if args.refine > 0:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation as R
        R_tgt = R.from_quat(TGT_CLEAR5[3:])

        def resid(q):
            s = fk7(q)
            dR = R_tgt * R.from_quat(s[3:]).inv()
            return np.concatenate([(s[:3] - TGT_CLEAR5[:3]) * 10.0, dR.as_rotvec()])

        d7 = np.linalg.norm(S - TGT_CLEAR5, axis=1)
        order = np.argsort(d7)[:args.refine]
        print(f"\n[精修] 从扫描最好的 {args.refine} 个位形继续（带限位）")
        best = None
        for k in order:
            try:
                r = least_squares(resid, Q[k], bounds=(lo, hi), max_nfev=4000)
            except Exception as e:
                print(f"  {e}")
                continue
            s = fk7(r.x)
            dp = float(np.linalg.norm(s[:3] - TGT_CLEAR5[:3]))
            dang = float(np.degrees((R_tgt * R.from_quat(s[3:]).inv()).magnitude()))
            print(f"  q={np.round(r.x,4).tolist()}  位置误差 {dp*1000:8.2f} mm  姿态误差 {dang:7.2f}°")
            if best is None or dp < best[0]:
                best = (dp, dang, r.x.copy())
        if best:
            print(f"\n[结论] clear5 帧0 在**带限位**下最好也只能到 {best[0]*1000:.1f} mm / {best[1]:.1f}°")
            if best[0] > 0.10:
                print("       ⇒ 该位姿不是'今天这份机器人 + 限位'能摆出来的。"
                      "clear5 采集时的模型/限位与现在不同。")


if __name__ == "__main__":
    main()
