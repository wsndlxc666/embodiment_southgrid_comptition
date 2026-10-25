"""姿态到底能不能摆出来？—— 只搜腕部两关节 (j5, j7) 的二维网格。

背景：用 ④ 自己的 OSC 伺服到训练帧 0 的**位置**能到 0.37mm，但**姿态**差 131.55°；
换成位置+姿态一起伺服，则位置停在 50mm、姿态 18.00°，且 j4/j6 顶在限位上。
于是结论一度是"这个起始姿态今天的模型摆不出来"。

但这个结论有个漏洞：位置与姿态不是独立的。实测
    j5 每 0.02rad 只让末端动 0.08mm，j7 只有 0.06mm —— **这两个关节几乎不改变位置**，
    只改变姿态（它们就在末端 site 的轴线上）。
所以正确做法是：先把位置锁死（用已验证能到 0.37mm 的那组 j1,j2,j3,j4,j6），
然后**单独扫 j5 / j7**，看姿态能不能对上。二维网格 5° 步长只要 5k 次采样。

判据：
  · 若网格里存在 位置误差 <20mm 且 姿态误差 <10° 的点 → 起始姿态**可达**，
    之前 131° 只是"位置+姿态同时伺服"被阻抗控制器带进了坏盆地，不是模型摆不出来。
  · 若整张网格姿态误差都 >60° → 确实摆不出来，那帧 0 的腕部朝向我们复现不了，
    只能靠改数据/重训，或者接受分布外。

第三阶段：用网格最优点当起点，再让 OSC 做一次 位置+姿态 伺服收尾，
因为阻抗控制器从好起点出发才收敛 —— 这也是将来 ④ 里要用的实现方式。

用法：.../bin/python probe_wrist_search.py
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--dataset", type=str,
                    default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--base_joints", type=str,
                    default="2.3709,0.3702,-0.8897,1.7229,-0.1114,-1.6065,1.043",
                    help="已知能精确到达目标位置的 j1..j7（位置只用 j1,j2,j3,j4,j6）")
    ap.add_argument("--step_deg", type=float, default=5.0)
    ap.add_argument("--tol_pos", type=float, default=0.020)
    ap.add_argument("--tol_rot", type=float, default=10.0)
    ap.add_argument("--polish_s", type=float, default=15.0)
    args = ap.parse_args()

    import pandas as pd
    from scipy.spatial.transform import Rotation as R

    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    st = np.stack(pd.read_parquet(fs[args.episode])["observation.state"].values)
    tgt = st[args.frame].astype(np.float64)
    p_t, q_t = tgt[7:10], tgt[10:14]
    R_t = R.from_quat(q_t)
    print(f"[目标] 帧{args.frame} 位置 {np.round(p_t,4).tolist()}  四元数 {np.round(q_t,4).tolist()}")

    q_base = np.array([float(x) for x in args.base_joints.split(",")], dtype=np.float64)
    assert q_base.size == 7, q_base

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(dataset_path="/tmp/_probe_scratch")
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=E.ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=scene_manager, frame_skip=5, orcagym_addr=args.orcagym_addr)
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE
    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")

    jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]
    l_idx = [env.joint(n) for n in E.agent_conf.l_arm["joint_names"]]

    def put(q7):
        d = {j: np.float32(0.0) for j in l_idx}
        d.update({j: np.float32(v) for j, v in zip(jidx, q7)})
        env.set_joint_qpos(d)
        env.mj_forward()

    def read():
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        dp = float(np.linalg.norm(s[7:10] - p_t))
        da = float(np.degrees((R_t * R.from_quat(s[10:14]).inv()).magnitude()))
        return dp, da, s

    put(q_base)
    dp0, da0, _ = read()
    print(f"[基线] 用给定关节组（位置已知可达）: 位置误差 {dp0*1000:.2f} mm  姿态误差 {da0:.2f}°")

    # ---- 二维网格：只动 j5 (=index 4) 和 j7 (=index 6)，其余锁死在 q_base ----
    lim = float(np.radians(177.6))
    vals = np.arange(-lim, lim + 1e-9, np.radians(args.step_deg))
    print(f"\n[网格] j5 × j7 各 {vals.size} 个取值（±177.6°，步长 {args.step_deg}°）"
          f" → {vals.size**2} 个采样点")
    t0 = time.time()
    best: list[tuple[float, float, float, float, np.ndarray]] = []
    allrows = []
    for a in vals:
        for b in vals:
            q = q_base.copy(); q[4] = a; q[6] = b
            put(q)
            dp, da, _ = read()
            allrows.append((dp, da, a, b))
            best.append((da, dp, a, b, q))
    el = time.time() - t0
    print(f"[网格] 完成 {len(allrows)} 点，用时 {el:.1f}s（{el/len(allrows)*1000:.1f} ms/点）")

    arr = np.array([(d[0], d[1]) for d in allrows])
    print(f"  位置误差范围 {arr[:,0].min()*1000:.1f} ~ {arr[:,0].max()*1000:.1f} mm")
    print(f"  姿态误差范围 {arr[:,1].min():.2f} ~ {arr[:,1].max():.2f} °")

    best.sort(key=lambda x: x[0])           # 按姿态误差排
    print(f"\n=== 姿态误差最小的 8 个点 ===")
    for da, dp, a, b, q in best[:8]:
        print(f"  姿态 {da:7.2f}°  位置 {dp*1000:7.1f} mm   j5={np.degrees(a):+7.1f}°  j7={np.degrees(b):+7.1f}°")

    okr = [t for t in best if t[0] < args.tol_rot]
    ok = [t for t in okr if t[1] < args.tol_pos]
    print(f"\n  姿态 <{args.tol_rot}° 的点：{len(okr)} 个；其中位置 <{args.tol_pos*1000:.0f}mm 的：{len(ok)} 个")
    if ok:
        da, dp, a, b, q = min(ok, key=lambda x: x[0] + x[1] / 0.02 * 2.0)
        print(f"  >>> 命中：姿态 {da:.2f}°  位置 {dp*1000:.1f} mm  "
              f"j = {np.round(np.degrees(q),2).tolist()}")
    else:
        print(f"  >>> 只靠 (j5,j7) 对不上。最好 {best[0][0]:.2f}° / {best[0][1]*1000:.1f} mm")

    # ---- 第三阶段：拿网格最好点当起点，OSC 做一次 位置+姿态 伺服 ----
    if args.polish_s > 0:
        da_b, dp_b, a_b, b_b, q_best = best[0]
        print(f"\n[收尾] 从 j5={np.degrees(a_b):+.1f}° j7={np.degrees(b_b):+.1f}° 出发，"
              f"OSC 位置+姿态 伺服 ≤{args.polish_s:.0f}s …")
        put(q_best)
        env.set_default_joint_values(E.build_default_joint_values())
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        for c in manager.controllers:
            c.reset()
        l_arm = E.create_arm(env, E.agent_conf.l_arm)
        r_arm = E.create_arm(env, E.agent_conf.r_arm)
        for _a2 in (l_arm, r_arm):
            _a2.controller.kp = np.ones(6, dtype=np.float64) * 150.0
            _a2.controller.kd = 2.0 * np.sqrt(_a2.controller.kp)
        l_grip = E.create_gripper(env, E.agent_conf.gripper_l)
        r_grip = E.create_gripper(env, E.agent_conf.gripper_r)
        for c in (l_arm, r_arm, l_grip, r_grip):
            manager.add_controller(c)
        t18 = np.zeros(18, dtype=np.float32); t18[:14] = tgt[:14]; t18[14:18] = tgt[14:18]
        tapply = E.action_dict_for_apply(E.parse_policy_action(t18))
        device = E.EEFDevice(l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                             lock_left_arm=True, grasp_integral=False, **tapply)
        manager.set_device(device)
        device.set_left_hold(tapply["l_pos_b"], tapply["l_quat_b"], tapply["l_grip_ctrl"])
        t1 = time.time(); last = 0.0
        while (time.time() - t1) < args.polish_s:
            device.set_target(**tapply)
            act = manager.run_controllers(); env.step(act); env.render()
            e = time.time() - t1
            if e - last >= 2.0:
                last = e
                dp, da, _ = read()
                print(f"    t={e:5.1f}s  位置 {dp*1000:7.2f} mm  姿态 {da:7.2f}°")
                if dp < 0.004 and da < 4.0:
                    break
        dp, da, s_fin = read()
        d = env.query_joint_qpos(jidx)
        qf = np.array([float(np.asarray(d[j]).reshape(-1)[0]) for j in jidx])
        print(f"\n===== 收尾结果 =====")
        print(f"  位置误差 {dp*1000:.2f} mm   姿态误差 {da:.2f}°")
        print(f"  关节 = {np.round(np.degrees(qf),2).tolist()} (度)")
        if dp < 0.006 and da < 10.0:
            print("  >>> 起始姿态可达。④ 里应当用『先摆到下面这组关节，再 OSC 收尾』的写法：")
            print(f"      base_joints = {np.round(qf,6).tolist()}")
        else:
            print("  >>> 仍未收敛；结合网格结论判断该姿态是否可达。")


if __name__ == "__main__":
    main()
