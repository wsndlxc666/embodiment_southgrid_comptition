"""仿真模型的运动链：从 `body_link1` 到 `ee_center_site_r` 之间到底夹了哪些关节？

已确认的事实（2026-09-17）：
  · `set_joint_qpos` 写回**成功**（直读 `_mjData.qpos` = 设定值，API 与直写一致）。
  · 同一组关节角，仿真 `mj_forward` 后 site 相对 base 的位置和离线 MJCF 的 FK 差 362~730mm。
  · 更狠的一条：**同一个 neutral 关节角**，probe_base_body 当初从仿真读到
    (0.6822, -0.0802, 0.2355)（与旧数据集帧 0 差 2e-4），今天这个探针读到
    (-0.075, -0.7752, 0.5857)。两次都对，那 base body 到末端之间就**不止那 7 个关节**。

离线 XML 里 body_link1 → body_link2(无关节) → arm_r_link1..7 → ee site，中间的关节恰好是
idx61..67。若仿真里这段链上还有别的关节（例如腰/躯干），那"只设 7 个臂关节 + 其它清零"
的离线 FK 就**根本不是同一个映射** —— 全部离线 IK 结论作废。

本脚本从 `gm._mjModel` 直接读链，逐段打印 body 与其携带的关节，并给出每个关节当前的
qpos 值，一眼看出哪些关节被离线 FK 忽略掉了。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/miniconda3/envs/orcalab/bin/python \
      src/examples/inference/g1_omnipicker/probe_sim_chain.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

SITE = "g1_omnipicker_ee_center_site_r"
BASE = "g1_omnipicker_body_link1"
JT = {0: "FREE", 1: "BALL", 2: "SLIDE", 3: "HINGE"}


def main() -> None:
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

    m, d = env.gym._mjModel, env.gym._mjData

    def bname(i):
        import mujoco
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body{i}>"

    def jname(i):
        import mujoco
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<jnt{i}>"

    sid = m.site(SITE).id
    sbody = m.site_bodyid[sid]
    bid = m.body(BASE).id

    print(f"site {SITE} 挂在 body = {bname(sbody)} (id={sbody})")
    print(f"base body {BASE} id={bid}\n")

    # 从 site body 向上走到 base，列出每一段的关节
    print("从 site 所在 body 向上到 base body 的链（打印顺序 = 根 → 叶）:")
    chain = []
    b = sbody
    while True:
        chain.append(b)
        if b == bid or b == 0:
            break
        b = m.body_parentid[b]
    for b in reversed(chain):
        jn = m.body_jntnum[b]
        js = []
        for k in range(jn):
            j = m.body_jntadr[b] + k
            qa = m.jnt_qposadr[j]
            js.append(f"{jname(j)}[{JT.get(m.jnt_type[j],'?')}] "
                      f"qposadr={qa} 当前qpos={np.asarray(d.qpos[qa:qa+1]).round(4).tolist()}")
        tag = "  <<< base" if b == bid else ""
        print(f"  {bname(b):44s} id={b:3d} 关节数={jn}{tag}")
        for s in js:
            print(f"        {s}")

    # base body 之上还有没有关节（含 free joint）？
    print("\nbase body 之上的父链:")
    b = m.body_parentid[bid]
    while b > 0:
        jn = m.body_jntnum[b]
        js = []
        for k in range(jn):
            j = m.body_jntadr[b] + k
            qa = m.jnt_qposadr[j]
            nd = 7 if m.jnt_type[j] == 0 else (4 if m.jnt_type[j] == 1 else 1)
            js.append(f"{jname(j)}[{JT.get(m.jnt_type[j],'?')}] "
                      f"qpos={np.asarray(d.qpos[qa:qa+nd]).round(4).tolist()}")
        print(f"  {bname(b):44s} id={b:3d} 关节数={jn}  {js}")
        b = m.body_parentid[b]

    # 结论判据
    arm = [f"idx{61+i}_arm_r_joint{i+1}" for i in range(7)]
    between = []
    for b in chain[:-1]:
        for k in range(m.body_jntnum[b]):
            j = m.body_jntadr[b] + k
            between.append(jname(j))
    extra = [n for n in between if not any(n.endswith(a) or a in n for a in arm)]
    print(f"\nbase → site 之间的关节共 {len(between)} 个:")
    print(f"  {between}")
    print(f"  其中**不属于**右臂 7 关节的: {extra if extra else '无'}")
    if extra:
        print("  ⇒ 离线 FK（只设 idx61..67、其余清零）与仿真不是同一个映射。")
        print("    必须把这些关节的当前值也喂进离线模型，或在仿真里做数值 IK。")
    else:
        print("  ⇒ 链上只有右臂 7 关节，离线 FK 的链结构是对的；差异应另找原因。")


if __name__ == "__main__":
    main()
