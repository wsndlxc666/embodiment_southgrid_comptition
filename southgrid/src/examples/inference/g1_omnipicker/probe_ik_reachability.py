"""训练帧 0 的右末端**位姿**到底可不可达？—— 离线全关节 IK（带限位 + 随机重启）。

为什么重做这件事：
  `probe_wrist_search.py` 只扫了 j5×j7 这**两个**关节，把 j1..j4,j6 全锁死在一组
  "位置解"上，结论是"最好 41.16°"。但 G1 右臂是 **7 自由度冗余臂** —— 同一个末端
  位姿通常有**一整族**关节解。只试一条分支就断言"摆不出来"，逻辑上不成立。
  而且那次的 OSC 收尾其实到了 50.6mm/17.88°，比网格的 41.16° 还好，本身就说明
  网格的结论不可信。

本脚本在 **MJCF 上做数值 IK**：阻尼最小二乘 + 关节限位投影 + 上千次随机重启，
覆盖冗余分支。

两个坑，都已处理：
  1. **四元数顺序**：state 第 10~14 维是 **xyzw**（`probe_quat_convention.py` 在仿真
     里实测：按 xyzw 解释残差 1.4e-08，按 wxyz 是 1.0）。scipy `R.from_quat` 默认
     就是 xyzw，直接用即可。**不要**因为 `query_site_pos_and_quat_B` 的 docstring
     写 `[w,x,y,z]` 就去重排 —— 那条 docstring 与实测不符。
  2. **雅可比**：`mj_jacSite` 给的 `jacr` 是**角速度**雅可比，直接配"旋转矢量误差"
     在误差大时增益是错的（会把可达误判成不可达）。这里改用**数值差分**，
     误差函数怎么定义雅可比就怎么算，7 个关节每步多 7 次 mj_forward，代价可接受。

判据：
  · 存在解 位置 <5mm 且 姿态 <5°  → 帧 0 位姿**可达**，④ 开局直接摆这组关节即可
    （纯配置、零训练），起始视觉就和训练完全一致。
  · 全局最好仍 >20°              → 确实不可达，只能回采集端改起始位姿或重训。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_ik_reachability.py
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

R_JOINTS = [f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)]
BASE_BODY = "body_link1"
EE_SITE = "ee_center_site_r"
XML = os.path.join(base_dir, "..", "..", "..", "model", "G1_omnipicker", "g1_omnipicker.xml")


def rotvec_of(Re: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 旋转矢量（模长 = 夹角，弧度）。"""
    c = np.clip((np.trace(Re) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2.0 * np.sin(ang)) * np.array(
        [Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=800)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol_pos", type=float, default=0.005)
    ap.add_argument("--tol_rot", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_every", type=int, default=25)
    args = ap.parse_args()

    import mujoco
    import pandas as pd
    from scipy.spatial.transform import Rotation as R

    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt = st[args.frame].astype(np.float64)
    p_star, q_star = tgt[7:10], tgt[10:14]
    # state 是 xyzw（仿真实测），scipy 默认也是 xyzw —— 直接喂
    R_star = R.from_quat(q_star).as_matrix()

    m = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in R_JOINTS]
    assert all(j >= 0 for j in jid), [n for n, j in zip(R_JOINTS, jid) if j < 0]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    lo = np.array([m.jnt_range[j][0] for j in jid])
    hi = np.array([m.jnt_range[j][1] for j in jid])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    assert sid >= 0 and bid >= 0

    print("右臂关节限位（度）:")
    for n, a, b in zip(R_JOINTS, lo, hi):
        print(f"  {n:24s} [{np.degrees(a):8.3f}, {np.degrees(b):8.3f}]")
    print(f"\n目标（数据集 ep{args.episode} 帧{args.frame}）")
    print(f"  位置   {np.round(p_star, 5).tolist()}")
    print(f"  四元数 {np.round(q_star, 5).tolist()}   (xyzw)")

    def fk(q7):
        d.qpos[:] = 0.0
        d.qpos[qadr] = q7
        mujoco.mj_forward(m, d)
        p = d.site_xpos[sid] - d.xpos[bid]
        Rb = d.xmat[bid].reshape(3, 3)
        Rs = d.site_xmat[sid].reshape(3, 3)
        return p, Rb.T @ Rs

    def err6(q7):
        p, Rc = fk(q7)
        return np.concatenate([p - p_star, rotvec_of(R_star.T @ Rc)]), p, Rc

    # ---- 约定 / 前向校验：已知位置解必须给 ~0 位置误差 ----
    q_known = np.clip(np.array([2.3709, 0.3702, -0.8897, 1.7229, -0.1114, -1.6065, 1.043]), lo, hi)
    p_k, R_k = fk(q_known)
    a_k = np.degrees(np.linalg.norm(rotvec_of(R_star.T @ R_k)))
    print(f"\n[校验] 已知位置解 base_joints → 位置误差 {np.linalg.norm(p_k - p_star)*1000:.2f} mm"
          f"   姿态误差 {a_k:.2f}°")

    rng = np.random.default_rng(args.seed)
    print(f"\n[IK] {args.restarts} 次随机重启 × ≤{args.iters} 步阻尼最小二乘（数值差分雅可比）…")
    print(f"[IK] 每 {args.report_every} 次重启报一次当前最好（**中途被杀也留得下结果**）", flush=True)
    t0 = time.time()
    best = (np.inf, np.inf, None)
    hits: list[tuple[float, float, np.ndarray]] = []
    # 每次重启的**收敛终点**都留着，后面按盆地去重 —— 单看"全局最好"会漏掉
    # 姿态一样好但不在关节限位上的分支。2026-09-17：全局最好那个解 j2、j6 双双顶限位，
    # 是贴边界的构型；而重启 25~50 区间出现过 16.79mm/0.20° 的解，明显是另一个盆地。
    # 顶限位的解在 OSC 下没有余量，所以值得把选择权留出来。
    finals: list[tuple[float, float, np.ndarray]] = []
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
                hits.append((dp, da, q.copy()))
                break
            J = np.zeros((6, 7))
            for i in range(7):
                qp = q.copy()
                # 顶上界时必须改**后向**差分：原写法 qp[i]=min(q+h,hi) 在 q[i]==hi[i]
                # 时得到 qp[i]==q[i]，除零 → 该列 NaN → 污染整个 LM 步（日志里的
                # "invalid value encountered in divide" 就是它）。
                qp[i] = q[i] + h if (q[i] + h) <= hi[i] else q[i] - h
                den = qp[i] - q[i]
                if abs(den) < 1e-12:
                    J[:, i] = 0.0
                    continue
                ep, _, _ = err6(qp)
                J[:, i] = (ep - e) / den
            A = J @ J.T + lam * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(A, e)
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
        if hits:
            break
        if (r + 1) % args.report_every == 0:
            print(f"  [{r+1:4d}/{args.restarts}] {time.time()-t0:6.1f}s  "
                  f"最好 {best[0]*1000:7.2f}mm / {best[1]:7.2f}°", flush=True)
    el = time.time() - t0
    print(f"  用时 {el:.1f}s")

    dp, da, q_best = best
    print(f"\n===== 全局最好 =====")
    print(f"  位置误差 {dp*1000:8.2f} mm   姿态误差 {da:8.2f}°")
    print(f"  关节（度）  = {np.round(np.degrees(q_best), 2).tolist()}")
    print(f"  关节（弧度）= {np.round(q_best, 6).tolist()}")
    sat = [R_JOINTS[i] for i in range(7)
           if min(abs(q_best[i] - lo[i]), abs(q_best[i] - hi[i])) < 1e-6]
    print(f"  顶在限位上的关节: {sat if sat else '无'}")

    if dp < args.tol_pos and da < args.tol_rot:
        print(f"\n>>> 帧 {args.frame} 的末端位姿**可达**。④ 开局直接摆这组关节，"
              f"起始视觉即与训练一致（纯配置、零训练）。")
        print(f"    ik_joints_r = {np.round(q_best, 6).tolist()}")
    else:
        print(f"\n>>> 仍未找到精确解（最好 {dp*1000:.1f}mm / {da:.1f}°）。"
              f"全关节 + 正确雅可比下仍不可达 → 帧 {args.frame} 的位姿摆不出来。")
    if hits:
        print(f"  （命中 {len(hits)} 个 <{args.tol_pos*1000:.0f}mm/<{args.tol_rot:.0f}° 的解）")

    # ---- 按盆地列出所有像样的解，标注限位余量 ----
    def _sat(q7):
        sat = [(R_JOINTS[i], np.degrees(min(q7[i] - lo[i], hi[i] - q7[i])))
               for i in range(7)
               if min(q7[i] - lo[i], hi[i] - q7[i]) < np.radians(2.0)]
        return sat

    ranked = sorted(finals, key=lambda t: (t[0] / args.tol_pos + t[1] / args.tol_rot))
    print(f"\n===== 按盆地去重后的候选解（共 {len(finals)} 次重启）=====")
    print(f"{'#':>3} {'位置mm':>8} {'姿态°':>8}  顶限位的关节（<2°余量）")
    shown = 0
    seen: list[np.ndarray] = []
    for dp_i, da_i, q_i in ranked:
        if shown >= 6:
            break
        if dp_i > 0.10 and da_i > 30.0:
            break
        # 盆地判据：关节空间里离已列出的解太近就跳过（不然前 6 名全是同一个盆地的副本）
        if any(np.linalg.norm(q_i - s) < np.radians(8.0) for s in seen):
            continue
        seen.append(q_i)
        sat = _sat(q_i)
        print(f"{shown+1:>3} {dp_i*1000:8.2f} {da_i:8.2f}  "
              f"{', '.join(f'{n}({m:+.2f}°)' for n, m in sat) if sat else '无'}")
        print(f"      关节(度) = {np.round(np.degrees(q_i), 2).tolist()}")
        shown += 1
    print("\n  选解建议：OSC 下顶限位的关节没有余量，优先选「姿态 <5° 且顶限位关节最少」的那个。")


if __name__ == "__main__":
    main()
