"""在**当前运行状态**下解右臂 IK，并且**写完再量一次**验证。

为什么不能沿用 start_ref_d22.json 里存好的 ik_joints_r：
  2026-09-17 实测 —— 把存好的那组关节原样写进 7 个关节，直读 qpos 完全等于设定值，
  但 base 系末端落在 [-0.0385, 0.2806, 0.4581]，离目标 [0.1585,-0.2792,0.4356] **593.82mm**
  （四条 FK 路径 —— 官方 query_site_pos_and_quat_B / 手算 / 清 free joint / 再 mj_forward ——
   给出完全相同的值，所以不是量法的问题）。
  而 probe_ik_sim.py 当初对同一组关节报的是 0.16mm。
  ⇒ 存下来的关节值不可信；**每次对齐时现场解**才稳。

本脚本：
  1. 用与运行时**完全同一条** FK 路径（body_link1 系，Rb.T@(spos-bp)）做 400 次随机重启 LM；
  2. 取最小限位余量足够的最优解；
  3. `set_joint_qpos` 写回去，再用官方 `query_site_pos_and_quat_B` 量一次，打印残差。
     第 3 步是硬验收：写不进去或写了不生效，这里就会暴露。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_ik_live.py [--restarts 400]
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
from run_task3_tool_storage_scored import ROBOT_BASE_BODY, EE_SITE_R  # noqa: E402

R_JOINTS = [f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)]
SR = os.path.join(base_dir, "start_ref_d22.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=400)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--tol_pos", type=float, default=0.005)
    ap.add_argument("--tol_rot", type=float, default=2.0)
    ap.add_argument("--min_margin_deg", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="把最优解写回 start_ref_d22.json")
    args = ap.parse_args()

    import mujoco
    from scipy.spatial.transform import Rotation as R
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

    with open(SR, "r", encoding="utf-8") as f:
        sr = json.load(f)
    p_t = np.asarray(sr["right_ee_pos_b"], dtype=np.float64).reshape(3)
    q_t = np.asarray(sr["right_ee_quat_b"], dtype=np.float64).reshape(4)
    R_star = R.from_quat(q_t).as_matrix()
    print(f"目标位置 {np.round(p_t, 5).tolist()}")

    m, d = env.gym._mjModel, env.gym._mjData
    jids = [m.joint(env.joint(n)).id for n in R_JOINTS]
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    sid = m.site(EE_SITE_R).id
    bid = m.body(ROBOT_BASE_BODY).id
    q0 = np.array(d.qpos).copy()

    def fk(q7):
        d.qpos[:] = q0
        d.qpos[qadr] = q7
        mujoco.mj_kinematics(m, d)
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        Rs = np.array(d.site_xmat[sid]).reshape(3, 3)
        return Rb.T @ (np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])), Rb.T @ Rs

    def rotvec(M):
        c = np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)
        a = float(np.arccos(c))
        if a < 1e-9:
            return np.zeros(3)
        return a / (2.0 * np.sin(a)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0],
                                                 M[1, 0] - M[0, 1]])

    def err6(q7):
        p, Rc = fk(q7)
        return np.concatenate([p - p_t, rotvec(R_star.T @ Rc)]), p, Rc

    def margin(q7):
        return float(np.degrees(np.min(np.minimum(q7 - lo, hi - q7))))

    rng = np.random.default_rng(args.seed)
    print(f"[IK] {args.restarts} 次重启 x <={args.iters} 步 LM（同一条 FK 路径）…", flush=True)
    t0 = time.time()
    best = (np.inf, np.inf, None)
    finals = []
    h = 1e-6
    for r in range(args.restarts):
        q = rng.uniform(lo, hi)
        lam = 1e-2
        e, p, Rc = err6(q)
        dp = float(np.linalg.norm(p - p_t))
        da = float(np.degrees(np.linalg.norm(rotvec(R_star.T @ Rc))))
        for _ in range(args.iters):
            if (dp, da) < (best[0], best[1]):
                best = (dp, da, q.copy())
            if dp < 1e-5 and da < 0.02:
                break
            J = np.zeros((6, 7))
            for i in range(7):
                qp = q.copy()
                qp[i] = q[i] + h if (q[i] + h) <= hi[i] else q[i] - h
                ep, _, _ = err6(qp)
                J[:, i] = (ep - e) / (qp[i] - q[i])
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            except np.linalg.LinAlgError:
                break
            qn = np.clip(q - dq, lo, hi)
            en, pn, Rcn = err6(qn)
            if np.linalg.norm(en) < np.linalg.norm(e):
                q, e, p, Rc = qn, en, pn, Rcn
                lam = max(lam * 0.7, 1e-6)
            else:
                lam = min(lam * 2.5, 1e3)
            dp = float(np.linalg.norm(p - p_t))
            da = float(np.degrees(np.linalg.norm(rotvec(R_star.T @ Rc))))
        finals.append((dp, da, q.copy()))
        if (r + 1) % 100 == 0:
            print(f"  [{r+1}/{args.restarts}] {time.time()-t0:5.1f}s 最好 {best[0]*1000:7.2f}mm "
                  f"/ {best[1]:6.2f}°", flush=True)

    print(f"\n用时 {time.time()-t0:.1f}s")
    print(f"全局最好 位置 {best[0]*1000:.2f}mm  姿态 {best[1]:.2f}°  余量 {margin(best[2]):.2f}°")
    ranked = sorted(finals, key=lambda t: t[0] / args.tol_pos + t[1] / args.tol_rot)
    seen, pick = [], None
    for dp_i, da_i, q_i in ranked:
        if any(np.linalg.norm(q_i - s) < np.radians(8.0) for s in seen):
            continue
        seen.append(q_i)
        mg = margin(q_i)
        print(f"  盆地#{len(seen)}  位置 {dp_i*1000:7.2f}mm  姿态 {da_i:6.2f}°  余量 {mg:6.2f}°  "
              f"{np.round(np.degrees(q_i), 2).tolist()}")
        if (pick is None and dp_i < args.tol_pos and da_i < args.tol_rot
                and mg > args.min_margin_deg):
            pick = (dp_i, da_i, q_i.copy(), mg)
        if len(seen) >= 6:
            break

    q_use = pick[2] if pick else best[2]
    print(f"\n选用：位置 {np.linalg.norm(fk(q_use)[0]-p_t)*1000:.2f}mm  "
          f"余量 {margin(q_use):.2f}°  关节(度) {np.round(np.degrees(q_use), 2).tolist()}")

    # ---- 硬验收：写回去，再用官方接口量 ----
    env.set_joint_qpos({env.joint(n): np.float32(v) for n, v in zip(R_JOINTS, q_use)})
    env.mj_forward()
    qq = env.query_site_pos_and_quat_B([EE_SITE_R], [ROBOT_BASE_BODY])[EE_SITE_R]
    p_meas = np.asarray(qq["xpos"], dtype=np.float64).reshape(3)
    q_meas = np.asarray(qq["xquat"], dtype=np.float64).reshape(4)[[1, 2, 3, 0]]
    ang = np.degrees(np.linalg.norm((R.from_quat(q_t).inv() * R.from_quat(q_meas)).as_rotvec()))
    print(f"\n>>> 写回后**官方接口**实测：位置残差 {np.linalg.norm(p_meas-p_t)*1000:.2f} mm"
          f"   姿态残差 {ang:.2f}°")
    print(f"    实测 base 系末端 = {np.round(p_meas, 5).tolist()}")

    if args.write and np.linalg.norm(p_meas - p_t) < 0.01 and ang < 3.0:
        sr["ik_joints_r"] = [float(x) for x in np.round(q_use, 6)]
        with open(SR, "w", encoding="utf-8") as f:
            json.dump(sr, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"    已写回 {SR} 的 ik_joints_r")
    else:
        print("    未写回（残差超限或没加 --write）")


if __name__ == "__main__":
    main()
