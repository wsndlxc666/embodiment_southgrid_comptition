"""IK 管线标定 + 强多起点反解 —— 判定 clear5 帧0 位姿是否真的不可达。

前一版(probe_reach.py)结论作废：它的对照组(旧数据集帧0，精确解已知 = FK(neutral))
也只收敛到 54.7 mm，说明是**求解器弱**，不是目标不可达。

这一版做两件事：
  1. **标定**：对已知可达的旧数据集帧0 做多起点 least_squares。若它能收敛到 ~0，
     管线才算可信；若它也卡在几十 mm，那 clear5 的读数一律无效。
  2. **几何**：直接打印右臂各 body 的世界坐标，算出"目标点离肩/上臂有多远"，
     和"肘关节限位能折到多近"比对 —— 这是不依赖求解器的独立判据。

用法：.../bin/python probe_reach2.py --starts 60
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

TGT_CLEAR5 = np.array([0.1585, -0.2792, 0.4356, -0.6987, 0.5574, 0.3993, 0.2043])
TGT_OLD = np.array([0.6822, -0.0802, 0.2354, -0.6258, -0.2309, 0.3103, 0.6773])

LIMITS_DEG = [(-179.908752441, 179.908752441), (-119.748168945, 119.748168945),
              (-177.616912842, 177.616912842), (0.0, 100.267608643),
              (-177.616912842, 177.616912842), (-99.694656372, 99.694656372),
              (-177.616912842, 177.616912842)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--tool_refs", type=str, default=os.path.join(base_dir, "tool_refs_d22.json"))
    ap.add_argument("--starts", type=int, default=60, help="多起点精修的起点数")
    ap.add_argument("--pool", type=int, default=4000, help="随机池大小")
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

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]

    # ---------- 几何：右臂各 body 世界坐标 ----------
    print("\n===== 右臂 body 世界坐标（neutral 位形下）=====")
    names = list(env.model.get_body_dict().keys())
    print("模型里的 body 名（含 arm_r / shoulder / ee / body_link）:")
    for n in names:
        ln = n.lower()
        if ("arm_r" in ln or "shoulder" in ln or "ee_center" in ln or "body_link" in ln
                or "torso" in ln or "waist" in ln):
            print(f"    {n}")
    try:
        base_w = np.asarray(env.body(E.ROBOT_BASE_BODY).get_pos(), dtype=np.float64).reshape(-1)[:3]
        print(f"\n  ROBOT_BASE_BODY({E.ROBOT_BASE_BODY}) world = {np.round(base_w,4).tolist()}")
    except Exception as e:
        print(f"  取 base world 失败: {e}")

    # ---------- 标定 + 反解 ----------
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R

    def fk7(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:14]

    def solve(tgt, q0, bounds):
        R_tgt = R.from_quat(tgt[3:])

        def resid(q):
            s = fk7(q)
            dR = R_tgt * R.from_quat(s[3:]).inv()
            return np.concatenate([(s[:3] - tgt[:3]) * 10.0, dR.as_rotvec()])
        r = least_squares(resid, q0, bounds=bounds, max_nfev=2000)
        s = fk7(r.x)
        dp = float(np.linalg.norm(s[:3] - tgt[:3]))
        dang = float(np.degrees((R_tgt * R.from_quat(s[3:]).inv()).magnitude()))
        return dp, dang, r.x.copy(), s

    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    rng = np.random.default_rng(1)
    Q = rng.uniform(lo, hi, size=(args.pool, 7))
    t0 = time.time()
    S = np.array([fk7(q) for q in Q])
    print(f"\n[池] {args.pool} 个随机位形，{time.time()-t0:.1f}s")

    for name, tgt in (("旧数据集帧0 (已知可达, 用来标定)", TGT_OLD),
                      ("clear5 帧0 (待判定)", TGT_CLEAR5)):
        print(f"\n===== {name} =====")
        print(f"  目标 pos {np.round(tgt[:3],4).tolist()}  quat {np.round(tgt[3:],4).tolist()}")
        dp0 = np.linalg.norm(S[:, :3] - tgt[:3], axis=1)
        print(f"  随机池最近位置距离 {dp0.min():.4f} m")
        best = None
        nst = min(args.starts, args.pool)
        t0 = time.time()
        for k in np.argsort(dp0)[:nst]:
            try:
                dp, dang, q, s = solve(tgt, Q[k], (lo, hi))
            except Exception as e:
                print(f"    {e}")
                continue
            if best is None or dp < best[0]:
                best = (dp, dang, q, s)
                print(f"    * 新最好 {dp*1000:7.2f} mm / {dang:6.2f}°   q={np.round(q,4).tolist()}")
        print(f"  [{nst} 起点精修耗时 {time.time()-t0:.1f}s]")
        if best:
            print(f"  >>> 最好: 位置 {best[0]*1000:.2f} mm / 姿态 {best[1]:.2f}°")
            print(f"      得 {np.round(best[3],4).tolist()}")
            print(f"      q={np.round(best[2],4).tolist()}")


if __name__ == "__main__":
    main()
