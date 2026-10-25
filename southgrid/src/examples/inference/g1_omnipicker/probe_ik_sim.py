"""在**仿真自己的模型**上解 IK —— 离线 MJCF 的 IK 解在仿真里无效。

2026-09-17 的排查链（每步都有实测）：
  1. `set_joint_qpos` 写回**成功**：直读 `_mjData.qpos` = 设定值，API 与直写一致。
  2. 链结构一致：base → site 之间恰好 7 个右臂关节，没有夹带别的关节。
  3. `_live_ee_b` 手算**没错**：official(`query_site_pos_and_quat_B`) 与手算差 0.00mm。
  4. 逐项比对模型常量：关节轴 / 限位 / body 位置 / site 偏移 **全部一致**，
     但**每一节臂 body 的四元数都不同**（例：`arm_r_link1` 仿真 180°绕 (0.909,0.4165,0)，
     离线 180°绕 y 轴）。⇒ 仿真加载的机器人不是 `model/G1_omnipicker/g1_omnipicker.xml`。
  5. 佐证：离线 XML 在 r_arm=neutral 下 FK = (0.6822,-0.0802,0.2355)，**正好等于旧数据集
     (single_grasp / single_place / tool) 帧 0**；而今天的仿真同关节给 (-0.075,-0.7752,0.5857)，
     与 clear5 的 (0.1585,-0.2792,0.4356) 也对不上。

所以离线 IK 的关节解在仿真里落不到目标上 —— 这正是 `--start_align ik` 残差 510mm 的原因。
本脚本把 IK 直接建在 `env.gym._mjModel` 上，解出来的关节值**在仿真里必然有效**。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_ik_sim.py
"""
from __future__ import annotations

import argparse
import glob
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


def rotvec_of(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2.0 * np.sin(ang)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--tol_pos", type=float, default=0.005)
    ap.add_argument("--tol_rot", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco
    import pandas as pd
    from scipy.spatial.transform import Rotation as R
    from yaml import Loader, load

    # ---- 目标：数据集帧 state[7:14] ----
    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt = st[args.frame].astype(np.float64)
    p_star, q_star = tgt[7:10], tgt[10:14]
    R_star = R.from_quat(q_star).as_matrix()
    print(f"目标（{os.path.basename(args.dataset)} ep{args.episode} 帧{args.frame}）")
    print(f"  位置   {np.round(p_star, 5).tolist()}")
    print(f"  四元数 {np.round(q_star, 5).tolist()}  (xyzw)")

    # ---- 起仿真，拿它自己的模型 ----
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

    m = env.gym._mjModel
    # 不能自己 new MjData(m)：这个模型（orcagym 的封装模型）会让 MjData 构造段错误。
    # 直接复用仿真自己的 _mjData —— 探针里把它当草稿纸用，反正之后会 reset。
    d = env.gym._mjData
    print(f"\n仿真模型: nq={m.nq} njnt={m.njnt} nbody={m.nbody}（复用仿真 _mjData）", flush=True)

    jids = [m.joint(env.joint(n)).id for n in R_JOINTS]
    print(f"[dbg] jids ok {jids}", flush=True)
    qadr = [m.jnt_qposadr[j] for j in jids]
    print(f"[dbg] qadr ok {qadr}", flush=True)
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    print(f"[dbg] limits ok", flush=True)
    sid = m.site(EE_SITE_R).id
    print(f"[dbg] sid={sid}", flush=True)
    bid = m.body(ROBOT_BASE_BODY).id
    print(f"[dbg] bid={bid}", flush=True)

    # 机器人自己的那个 free joint：把它摆到原点 + 单位四元数，
    # 这样 base 坐标系 == 世界坐标系，口径与离线 FK 一致。
    q0 = np.array(d.qpos).copy()
    robot_free_jid = m.joint(env.joint("free_joint")).id
    robot_free_adr = int(m.jnt_qposadr[robot_free_jid])
    print(f"[dbg] 机器人 free joint adr={robot_free_adr} 当前 qpos="
          f"{np.round(q0[robot_free_adr:robot_free_adr+7], 4).tolist()}", flush=True)

    def fk(q7, _dbg=False):
        if _dbg:
            print("   [fk] 写 qpos", flush=True)
        d.qpos[:] = q0                          # 先恢复现场，别把 9 个自由物体叠到原点
        d.qpos[robot_free_adr:robot_free_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[qadr] = q7
        if _dbg:
            print("   [fk] mj_kinematics", flush=True)
        # 只用 mj_kinematics：它照样更新 xpos/xmat/site_xpos/site_xmat，
        # 但不做碰撞与动力学 —— IK 里几十万次调用，既快又不会因为穿模崩掉。
        mujoco.mj_kinematics(m, d)
        if _dbg:
            print("   [fk] 读位姿", flush=True)
        p = np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        Rs = np.array(d.site_xmat[sid]).reshape(3, 3)
        return p, Rb.T @ Rs

    print("[dbg] fk 定义完成，试调一次…", flush=True)
    _pt, _ = fk(np.zeros(7), _dbg=True)
    print(f"[dbg] fk(0) = {np.round(_pt, 5).tolist()}  没崩", flush=True)

    def err6(q7):
        p, Rc = fk(q7)
        return np.concatenate([p - p_star, rotvec_of(R_star.T @ Rc)]), p, Rc

    # 校验：neutral 下仿真模型给出的末端，应当等于旧数据集帧 0 的那个值吗？
    q_neu = np.array(E.agent_conf.r_arm["neutral_joint_values"])
    p_n, _ = fk(q_neu)
    print(f"\n[对照] 仿真模型 r_arm=neutral → 相对 base {np.round(p_n, 5).tolist()}")
    print(f"       旧数据集帧0 右臂位置 = [0.6822, -0.0802, 0.2354]")
    print(f"       ⇒ 差 {np.linalg.norm(p_n - np.array([0.6822, -0.0802, 0.2354]))*1000:.1f} mm"
          f"（若为 0 说明仿真模型就是旧的那份）")

    rng = np.random.default_rng(args.seed)
    print(f"\n[IK] 在**仿真模型**上 {args.restarts} 次重启 × ≤{args.iters} 步 LM…", flush=True)
    t0 = time.time()
    best = (np.inf, np.inf, None)
    finals: list = []
    h = 1e-6
    for r in range(args.restarts):
        q = rng.uniform(lo, hi)
        lam = 1e-2
        e, p, Rc = err6(q)
        dp = float(np.linalg.norm(p - p_star))
        da = float(np.degrees(np.linalg.norm(rotvec_of(R_star.T @ Rc))))
        for _ in range(args.iters):
            if (dp, da) < (best[0], best[1]):
                best = (dp, da, q.copy())
            if dp < args.tol_pos and da < args.tol_rot:
                break
            J = np.zeros((6, 7))
            for i in range(7):
                qp = q.copy()
                qp[i] = q[i] + h if (q[i] + h) <= hi[i] else q[i] - h
                den = qp[i] - q[i]
                if abs(den) < 1e-12:
                    continue
                ep, _, _ = err6(qp)
                J[:, i] = (ep - e) / den
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
            dp = float(np.linalg.norm(p - p_star))
            da = float(np.degrees(np.linalg.norm(rotvec_of(R_star.T @ Rc))))
        finals.append((dp, da, q.copy()))
        # 不提前 break：多跑几个盆地，才能挑出「不顶限位」的那个
        # （OSC 下顶限位的关节没有余量，会被一直压在边界上）。
        if (r + 1) % 200 == 0:
            print(f"  [{r+1:4d}/{args.restarts}] {time.time()-t0:6.1f}s 最好 "
                  f"{best[0]*1000:7.2f}mm / {best[1]:7.2f}°", flush=True)

    dp, da, q_best = best
    print(f"  用时 {time.time()-t0:.1f}s")
    print(f"\n===== 仿真模型上的全局最好 =====")
    print(f"  位置误差 {dp*1000:8.2f} mm   姿态误差 {da:8.2f}°")
    print(f"  关节(度)   = {np.round(np.degrees(q_best), 2).tolist()}")
    print(f"  关节(弧度) = {np.round(q_best, 6).tolist()}")
    sat = [(R_JOINTS[i], float(np.degrees(min(q_best[i]-lo[i], hi[i]-q_best[i]))))
           for i in range(7) if min(q_best[i]-lo[i], hi[i]-q_best[i]) < np.radians(2.0)]
    print(f"  顶限位的关节(<2°余量): {[(n, round(m_,2)) for n, m_ in sat] if sat else '无'}")

    if dp < args.tol_pos and da < args.tol_rot:
        print(f"\n>>> 可达。这组关节在**仿真里**就是训练帧 {args.frame} 的起始位姿：")
        print(f"    ik_joints_r = {np.round(q_best, 6).tolist()}")
    else:
        print(f"\n>>> 仿真模型上仍不可达（最好 {dp*1000:.1f}mm / {da:.1f}°）。"
              f"那说明 clear5 采集用的模型与今天的仿真模型也不一样 —— 需要另找模型。")

    def margin(q7):
        return float(np.degrees(np.min(np.minimum(q7 - lo, hi - q7))))

    ranked = sorted(finals, key=lambda t: t[0] / args.tol_pos + t[1] / args.tol_rot)
    print(f"\n===== 按盆地去重后的候选解（共 {len(finals)} 次重启的终点）=====")
    print(f"{'#':>3} {'位置mm':>9} {'姿态°':>8} {'最小限位余量':>12}   关节(度)")
    seen: list = []
    shown = 0
    clean = None
    for dp_i, da_i, q_i in ranked:
        if shown >= 8 or (dp_i > 0.10 and da_i > 30.0):
            break
        if any(np.linalg.norm(q_i - s) < np.radians(8.0) for s in seen):
            continue
        seen.append(q_i)
        mg = margin(q_i)
        if clean is None and dp_i < args.tol_pos and da_i < args.tol_rot and mg > 10.0:
            clean = (dp_i, da_i, q_i.copy(), mg)
        print(f"{shown+1:>3} {dp_i*1000:9.2f} {da_i:8.2f} {mg:11.2f}°   "
              f"{np.round(np.degrees(q_i), 2).tolist()}")
        shown += 1

    print()
    if clean is not None:
        dp_c, da_c, q_c, mg_c = clean
        print(f">>> 推荐（满足 <{args.tol_pos*1000:.0f}mm/<{args.tol_rot:.0f}° 且最小限位余量 "
              f"{mg_c:.2f}°）：{dp_c*1000:.2f}mm / {da_c:.2f}°")
        print(f"    ik_joints_r = {np.round(q_c, 6).tolist()}")
        print(f"    关节(度)    = {np.round(np.degrees(q_c), 2).tolist()}")
    else:
        print(">>> 没有余量 >10° 的精确解，只能在上面几个里挑余量最大的。")


if __name__ == "__main__":
    main()
