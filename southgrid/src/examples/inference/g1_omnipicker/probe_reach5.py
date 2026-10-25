"""可靠的覆盖度 + 精修判据：训练帧位姿今天到底够不够得到。

probe_reach4 用"20 个随机起点 + 位置-only 精修"对 15 帧全部失败(130~286mm)。
但那可能是**求解器的锅**：3 维目标、7 个未知量、随机起点极易落进劣质盆地。
（证据：probe_reach3 里对已知可达的对照组，只有"从 neutral 起"那一次能到 0.062mm，
 40 个随机起点都没能改进它。）

这一版换成稳健的两步：
  1. **覆盖度**：在关节限位内均匀撒 N 个位形，得到末端位置点云。对每个训练帧，
     量"它离点云最近有多远"。这个量**不依赖任何优化器**；如果训练帧落在点云
     覆盖范围内，最近距离会是十几毫米量级（受采样密度限制），而不是 130mm。
  2. **精修**：拿离训练帧最近的那个点云位形当起点做位置-only 反解。
     起点已经在正确盆地附近，收敛才有意义。

两件事同时看：覆盖度说"够不够得到"，精修说"能不能精确命中"。

用法：.../bin/python probe_reach5.py --n 20000
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
    ap.add_argument("--frames", type=str,
                    default="0,50,100,150,300,600,900,1100,1200,1500,1700,1800,1900,2000,2100")
    ap.add_argument("--n", type=int, default=20000, help="点云采样数")
    ap.add_argument("--seed", type=int, default=3)
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
    neutral = np.asarray(E.agent_conf.r_arm["neutral_joint_values"], dtype=np.float64)

    def fk_pos(q):
        env.set_joint_qpos({j: np.float32(v) for j, v in zip(jidx, q)})
        env.mj_forward()
        storage.reset_cycle()
        s = np.asarray(storage.build_state(storage.obs_callback(env)), dtype=np.float64).reshape(-1)
        return s[7:10]

    lo = np.radians([a for a, _ in LIMITS_DEG])
    hi = np.radians([b for _, b in LIMITS_DEG])

    print(f"[点云] 采样 {args.n} 个位形 …")
    rng = np.random.default_rng(args.seed)
    Q = rng.uniform(lo, hi, size=(args.n, 7))
    t0 = time.time()
    P = np.empty((args.n, 3))
    for i in range(args.n):
        P[i] = fk_pos(Q[i])
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{args.n}  {time.time()-t0:.1f}s")
    print(f"[点云] 完成 {time.time()-t0:.1f}s")
    print(f"  点云 bbox  x[{P[:,0].min():.4f},{P[:,0].max():.4f}]"
          f" y[{P[:,1].min():.4f},{P[:,1].max():.4f}] z[{P[:,2].min():.4f},{P[:,2].max():.4f}]")
    print(f"  训练  bbox  x[{st[:,7].min():.4f},{st[:,7].max():.4f}]"
          f" y[{st[:,8].min():.4f},{st[:,8].max():.4f}] z[{st[:,9].min():.4f},{st[:,9].max():.4f}]")

    # 采样密度参考：点云自身的最近邻距离尺度（取 200 个点的中位数）
    idx = rng.choice(args.n, size=min(200, args.n), replace=False)
    nn_self = []
    for k in idx:
        d = np.linalg.norm(P - P[k], axis=1)
        d[k] = np.inf
        nn_self.append(d.min())
    print(f"  点云自身最近邻中位数 = {np.median(nn_self)*1000:.1f} mm （采样分辨率上限）")

    from scipy.optimize import least_squares
    frames = [int(x) for x in args.frames.split(",") if x.strip()]

    # 额外并入：conf neutral 的 FK 位置（= ④ 现在的起始位姿）
    p_neutral = fk_pos(neutral)
    extra = [("conf neutral(④起始)", p_neutral)]
    print(f"\n  [参考] conf neutral FK 位置 = {np.round(p_neutral,4).tolist()}")

    print(f"\n===== 逐帧覆盖度 + 从最近点精修 =====")
    print(f"{'帧':>6s} {'训练位置':>28s} {'离点云最近':>10s} {'精修后':>10s}  结论")
    rows = []
    for fr in frames:
        if fr >= len(st):
            continue
        tgt = st[fr, 7:10]
        d = np.linalg.norm(P - tgt, axis=1)
        k = int(d.argmin())
        cov = float(d[k])
        r = least_squares(lambda q: fk_pos(q) - tgt, Q[k], bounds=(lo, hi),
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
        dref = float(np.linalg.norm(fk_pos(r.x) - tgt))
        rows.append((fr, cov, dref))
        verdict = "✓ 命中" if dref < 0.005 else ("△ 接近" if dref < 0.03 else "✗ 够不到")
        print(f"{fr:>6d} {str(np.round(tgt,4).tolist()):>28s} {cov*1000:>7.1f} mm {dref*1000:>7.1f} mm  {verdict}")

    for name, tgt in extra:
        d = np.linalg.norm(P - tgt, axis=1)
        k = int(d.argmin())
        cov = float(d[k])
        r = least_squares(lambda q: fk_pos(q) - tgt, Q[k], bounds=(lo, hi),
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=800)
        dref = float(np.linalg.norm(fk_pos(r.x) - tgt))
        print(f"{name:>22s} {str(np.round(tgt,4).tolist()):>28s} {cov*1000:>7.1f} mm {dref*1000:>7.1f} mm")

    hit = sum(1 for _, _, d in rows if d < 0.005)
    near = sum(1 for _, _, d in rows if d < 0.03)
    print(f"\n命中(<5mm) {hit}/{len(rows)} ；接近(<30mm) {near}/{len(rows)}")
    if hit + near >= max(2, len(rows) // 2):
        print("⇒ 训练位姿今天的模型**能摆出来**（是采样/优化问题，不是模型差异）。")
    else:
        print("⇒ 训练位姿今天的模型**摆不出来**：模型确实变了，"
              "或点云/精修仍不够强 —— 看'离点云最近'那一列是否远大于采样分辨率。")


if __name__ == "__main__":
    main()
