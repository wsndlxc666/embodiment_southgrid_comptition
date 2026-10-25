"""干净的 IK 判据：先只解位置（3 残差 / 7 未知量），再管姿态。

前两版都作废，原因：`probe_reach2.py` 里对**已知可达**的旧数据集帧0
（精确解 = FK(conf neutral)，距离 0.0004）多起点 least_squares 也只收敛到 47 mm。
姿态残差量级(~2.7 rad)压过位置残差(×10 后 ~0.6)，求解器在 6/7 的欠定问题里
被姿态拖着走，落进别的盆地。所以那两版的"不可达"读数一律无效。

这一版：
  1. 平滑性自检 —— FK 对单个关节 ±0.02 rad 的响应是否连续、量级合理。
  2. 位置-only 反解（scipy least_squares，3 残差），大量随机起点。
     对照组必须能到 ~0；到了才说明管线可用。
  3. 两个目标都用位置-only 判定"够不够得到"，再对能到的那个补姿态。

用法：.../bin/python probe_reach3.py
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
    ap.add_argument("--starts", type=int, default=40)
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

    jn = list(E.agent_conf.r_arm["joint_names"])
    jidx = [env.joint(n) for n in jn]
    neutral = np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64)
    print("[关节] " + ", ".join(f"{n}→{i}" for n, i in zip(jn, jidx)))

    def fk7(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:14]

    # ---------- 1. 平滑性自检 ----------
    print("\n===== 平滑性自检（neutral 处逐关节 ±0.02 rad）=====")
    s0 = fk7(neutral)
    print(f"  FK(neutral) = {np.round(s0,4).tolist()}")
    print(f"  与 TGT_OLD 距离 = {np.linalg.norm(s0-TGT_OLD):.5f}  (应≈0，因为 conf neutral 就是它的解)")
    for i in range(7):
        for sgn in (+1, -1):
            q = neutral.copy(); q[i] += sgn * 0.02
            ds = fk7(q) - s0
            print(f"    j{i+1}{sgn:+d}0.02rad → Δpos {np.round(ds[:3],5).tolist()}  |Δpos|={np.linalg.norm(ds[:3])*1000:6.2f} mm")

    # ---------- 2/3. 位置-only 反解 ----------
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    def solve_pos(tgt_pos, q0):
        r = least_squares(lambda q: fk7(q)[:3] - tgt_pos, q0, bounds=(lo, hi),
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
        return float(np.linalg.norm(fk7(r.x)[:3] - tgt_pos)), r.x.copy()

    rng = np.random.default_rng(7)

    for name, tgt in (("旧数据集帧0 (对照组, 已知可达)", TGT_OLD),
                      ("clear5 帧0 (待判定)", TGT_CLEAR5)):
        print(f"\n===== {name}  pos {np.round(tgt[:3],4).tolist()} =====")
        # 从 neutral 直接解（对照组必然为 0）
        d, q = solve_pos(tgt[:3], neutral)
        print(f"  从 neutral 起解 → 位置误差 {d*1000:8.3f} mm")
        best = (d, q)
        t0 = time.time()
        for k in range(args.starts):
            q0 = rng.uniform(lo, hi, 7)
            d, q = solve_pos(tgt[:3], q0)
            if d < best[0]:
                best = (d, q)
                print(f"    * 新最好 {d*1000:8.3f} mm  q={np.round(q,4).tolist()}")
        print(f"  [{args.starts} 随机起点，{time.time()-t0:.1f}s]")
        d, q = best
        print(f"  >>> 位置-only 最好 {d*1000:.3f} mm   q={np.round(q,4).tolist()}")
        s = fk7(q)
        try:
            dang = float(np.degrees((R.from_quat(tgt[3:]) * R.from_quat(s[3:]).inv()).magnitude()))
        except Exception:
            dang = float("nan")
        print(f"      该解姿态误差 {dang:.2f}°（先不管，位置够得到就已经说明可达）")
        if d < 0.005 and dang > 5.0:
            # 位置到位了，现在在位置解附近补姿态
            R_tgt = R.from_quat(tgt[3:])
            r = least_squares(
                lambda qq: np.concatenate([
                    (fk7(qq)[:3] - tgt[:3]) * 50.0,
                    (R_tgt * R.from_quat(fk7(qq)[3:]).inv()).as_rotvec() * 0.2]),
                q, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=2000)
            s2 = fk7(r.x)
            dp2 = float(np.linalg.norm(s2[:3] - tgt[:3]))
            da2 = float(np.degrees((R_tgt * R.from_quat(s2[3:]).inv()).magnitude()))
            print(f"      [补姿态] 位置 {dp2*1000:.3f} mm / 姿态 {da2:.3f}°   q={np.round(r.x,4).tolist()}")


if __name__ == "__main__":
    main()
