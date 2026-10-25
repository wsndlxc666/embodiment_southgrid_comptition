"""Task3 电力工具归纳收置 · 评分接入驱动（g1_omnipicker）

在单个评分 attempt（task3_tool_storage，本地 scorer 180s 硬时限）内，用 OpenPI
工具策略依次以训练 prompt 抓取 5 件工具放入工具箱；复用 eval 脚本的控制器/
相机/策略管线（左臂锁定，右臂 OSC）。由 orca_scorer_client 的 ScorerClient
自动评分（本地 scorer 采样 50051 场景 + 录屏上传中央服务器）。

工具是场景中的静态 free-joint 物体，env.reset()/update_scene() 不会复位它们，
因此不重置场景、只切换 prompt 连续收纳，物理全程连续。

用法（务必用 orcalab 的绝对路径 python，避免 openpi venv 抢占解释器）：
    # 0) 只读探测当前场景（不连相机/不评分/不动工具）
    .../bin/python run_task3_tool_storage_scored.py --dry-run

    # 1) 正式评分
    .../bin/python run_task3_tool_storage_scored.py \
        --team-id csgComp_Team_08 --team-token <你的队伍token>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import eval_g1_omnipicker_tool_lerobot as E  # noqa: E402

# ---------------------------------------------------------------------------
# Task3 评分常量（与 configs/tasks.yaml 一致）
# ---------------------------------------------------------------------------
TOOLBOX_SITE = "Group_Interactive_ToolBox_base_site"
# 阶段顺序 = **训练数据的顺序**（扳手→螺丝刀→电工刀(左)→手电筒→电工刀(右)），
# 不是官方 step_id 在评分器里出现的顺序。
#
# 为什么改（2026-09-16，用户确认）：**顺序不算分**。评分器只判"每件工具最终有没有进箱"
# （6 个 step 各自独立给分，weight 5+7×5），开局摆放和先后完全由我们定。
# 而官方 SCORE_PHASES 的顺序（电工刀(右)→螺丝刀→扳手→手电筒→电工刀(左)）和训练数据
# **完全不同** —— 原来的顺序等于每轮都在逼模型做它从没学过的次序，纯属自伤。
# 按训练顺序跑，模型只需要"按它学过的那条时间线往前走"。
#
# 注：每阶段的 `prompt` 仍是裸工具名，而训练时的 task 串**从来不含工具名**
# （只有"把工具都收进工具箱"等 3 句）。所以这几个 prompt 对模型基本是分布外的，
# 之前实测换 prompt 只动 2~4 个 SEM。留在这里只是为了日志/分段可读。
# 要不要改成训练 prompt，见 README 二·八。
SCORE_PHASES: list[dict] = [
    {"step_id": "pick_wrench",            "site": "Group_Interactive_Spanner_task_spanner_site",                        "prompt": "扳手"},
    {"step_id": "pick_screwdriver",       "site": "Group_Interactive_Screwdriver_task_screwdriver_site",                "prompt": "螺丝刀"},
    {"step_id": "pick_electrician_knife", "site": "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_site", "prompt": "电工刀(左)"},
    {"step_id": "pick_flashlight",        "site": "Group_Interactive_Flashlight_task_flashlight_site",                  "prompt": "手电筒"},
    {"step_id": "pick_voltage_pen",       "site": "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_site", "prompt": "电工刀(右)"},
]
# 单回合模式（默认）用的 prompt。训练语料里只有 3 句、全不含工具名，轮换出现：
#   "把工具全部收进箱子里"(29集) / "把桌面上的工具都收进工具箱"(40集) / "把所有工具从左到右放回工具箱"(31集)
# 选第 3 句是因为它同时点明"所有工具"和"从左到右"——正是我们想让模型照做的两件事。
# 三句在训练里近等比例出现，所以任选一句都在分布内；要换用 --episode_prompt。
EPISODE_PROMPT = "把所有工具从左到右放回工具箱"
ROBOT_BASE_BODY = "g1_omnipicker_body_link1"
# 右手指末端 site（用于诊断：区分"伸错位置 / 够不到 / 抓不住"）
EE_SITE_R = "g1_omnipicker_ee_center_site_r"
# 触发"已放入"的距离阈值（工具 site 距 toolbox site），m
PLACED_DIST = 0.20
# 右夹爪内爪电机目标值低于此视为张开/释放（-0.856 附近为张开，+值越大越闭合）
GRIP_OPEN_MOTOR = -0.35
# 释放确认保持时间
CONFIRM_S = 1.2
# 评分 attempt 软时限（本地 scorer 硬限 180s），预留 finish+上传。
# ⚠ 这是**评分路径**的默认值，也是唯一的"来源常量"；**生效值**放在下面的
#   SOFT_DEADLINE_S（模块全局），由 _resolve_soft_deadline() 在解析完参数后覆盖。
#   **两者必须分开**：若让 SOFT_DEADLINE_S 兼任来源，第二次解析参数就会读到
#   上一次写进去的 inf，评分路径会静默变成不限时。
#   inf 参与比较时行为就是"永不触发"，所以所有 `< SOFT_DEADLINE_S` 的写法都不用改。
SCORING_DEADLINE_S = 165.0
SOFT_DEADLINE_S = SCORING_DEADLINE_S          # 生效值（main 里按参数覆盖）


def _resolve_soft_deadline(no_score: bool, override: float | None) -> float:
    """算出生效的挂钟软时限；返回 float("inf") 表示**不限时**。

    为什么不设时限在评分时是危险的：本地 scorer 硬限 180s，超了它会直接掐掉
    attempt（录屏/上传一起废），所以评分路径必须留着 165s。
    而演练（--no-score）没有那个硬限，165s 又只买到 ~32s 仿真（action_repeat=1
    下 20 步推理才 0.1s 仿真），五件工具在训练集里要 ~228s 仿真 —— 差 7 倍。
    于是演练里 165s 一定到点、永远等不到"五件全收完"那个真终止条件。
    所以：**演练默认不限时**，只按『五件全部入箱』或手动 Ctrl-C 停。

    override: None=按上面的默认自动选；0=强制不限时；>0=用这个秒数。
    """
    if override is None:
        return float("inf") if no_score else SCORING_DEADLINE_S
    if override <= 0.0:
        return float("inf")
    return float(override)


def _deadline_text(dl: float) -> str:
    """给人看的时限文字（inf 直接印 inf 会让人以为是 bug）。"""
    return "不限时" if not np.isfinite(dl) else f"{dl:.0f}s"


def _loop_exit_reason(dl: float) -> str:
    """主循环既不是被 done_reason 跳出、也不是被 Ctrl-C 打断时，是怎么退出的。

    不限时的情况下这个分支**只可能是手动中断或异常**，所以不能再说'软时限到'
    —— 2026-09-20 就是被这句错误的兜底文案误导，把一次 165s 到点读成了"任务做完了"。
    """
    return "软时限到（循环条件退出）" if np.isfinite(dl) else "手动停止 / 循环意外退出"

# ---------------------------------------------------------------------------
# 工具初始摆放：复刻采集器的"种回标称槽位"（--plant_layout）
#
# 来源（SouthGrid/src/examples/dataCollection/g1_omnipicker/）：
#   标称表  g1_omnipicker_collection_scripted_tool_lerobot.py:74-102
#   实现    g1_omnipicker_collect_clear5_lerobot.py:340-363  _plant_tools
#   校验    g1_omnipicker_collect_clear5_lerobot.py:382-395  _validate_layout
#
# 采集器**每一集**的顺序是 reset → update_scene → set_default_joint_values →
# _plant_tools → _settle(180) → set_default + mj_forward，然后才开始录帧。
# 而工具是 free-joint 物体，reset()/update_scene() 都**不复位**它们（见本文件开头
# 的旧注释）—— 于是 ④ 一直沿用 ORCA 关卡自带的摆放。实测（d22_ik8 开局探测）：
# 手电筒在箱边 0.084m，其余 4 件 0.355/0.409/0.474/0.799m。官方判据是
# "离箱心 ≤0.25m 且持续 1.0s"，于是开局就白送一件的分（6.67/35），
# cycle_idx 还会开局 +1，把模型推到"去抓螺丝刀"的 state 上。
#
# 真正的要害：训练里 5 件**全在桌面**（标称槽位 y = -0.069 … -0.518），
# 第 4 轮才走到 TOOL_REF[3] 抓手电筒（离箱 0.589m）。模型是不看图的开环
# 复读机（probe_leftarm_causal.py 实测视觉贡献 < 噪声底），到点会发现那儿是空的。
# 所以评测必须把 5 件种回标称槽位。
# ---------------------------------------------------------------------------
TOOL_JOINTS = [
    "Group_Interactive_Spanner_task_spanner_joint",
    "Group_Interactive_Screwdriver_task_screwdriver_joint",
    "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
    "Group_Interactive_Flashlight_task_flashlight_joint",
    "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
]
# 顺序与 TOOL_JOINTS / SCORE_PHASES 一致
TOOL_LABELS = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]
# base 系标称槽位（≈部署默认摆放），z 由表给出
TOOL_REF_POS_B = np.asarray(
    [
        [0.5654831, -0.0693993, 0.1514528],
        [0.5599098, -0.1873288, 0.1568153],
        [0.5764828, -0.3049245, 0.1447767],
        [0.5079708, -0.4143662, 0.1536523],
        [0.5802917, -0.5180783, 0.1439366],
    ],
    dtype=np.float64,
)
TOOL_REF_QUAT_XYZW_B = np.asarray(
    [
        [-0.0, 0.7071068, 0.7071067, 0.0],
        [0.4999249, -0.5000751, -0.4999244, 0.5000756],
        [-0.0, 0.7071065, 0.7071071, 0.0],
        [-0.5, 0.5, -0.4999995, 0.5000005],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
# 采集器的落点抖动默认值：x±8mm y±18mm yaw±6°，另有 12% 概率某一件被"碰歪"到 y±40mm/yaw±14°。
# 评测默认**不抖**（纯标称=训练分布的中心），要测鲁棒性再用 --plant_jitter 打开。
TOOL_JITTER_TRAIN = dict(j_x=0.008, j_y=0.018, j_yaw_deg=6.0,
                         big_frac=0.12, big_y=0.040, big_yaw_deg=14.0)


def _sample_tool_layout(rng, j_x, j_y, j_yaw_deg, big_frac, big_y, big_yaw_deg):
    """采集器 `_sample_clear_layout` 的等价物：返回 (xy, yaw)，键是工具下标。"""
    big_idx = int(rng.integers(0, 5)) if (big_frac > 0 and rng.random() < big_frac) else -1
    xy, yaw = {}, {}
    for i in range(5):
        dx = float(rng.uniform(-j_x, j_x))
        if i == big_idx:
            dy = float(rng.uniform(-big_y, big_y))
            dyaw = float(np.deg2rad(rng.uniform(-big_yaw_deg, big_yaw_deg)))
        else:
            dy = float(rng.uniform(-j_y, j_y))
            dyaw = float(np.deg2rad(rng.uniform(-j_yaw_deg, j_yaw_deg)))
        xy[i] = (float(TOOL_REF_POS_B[i, 0] + dx), float(TOOL_REF_POS_B[i, 1] + dy))
        yaw[i] = dyaw
    return xy, yaw


def _plant_tools(env, xy, yaw) -> None:
    """把 5 件工具种回 (标称+抖动) 位姿：自由关节 qpos 直接写世界位姿。

    与 g1_omnipicker_collect_clear5_lerobot.py:_plant_tools 逐行对应。
    """
    from scipy.spatial.transform import Rotation as _Rq
    # 必须传全名 ROBOT_BASE_BODY：agent_conf.base_body 是短名 "body_link1"，会 KeyError。
    # 姿态走四元数（不用矩阵），与采集器取值路径逐字节一致。
    bpos, _bmat, bquat_wxyz = env.get_body_xpos_xmat_xquat([ROBOT_BASE_BODY])
    base_pos = np.asarray(bpos, dtype=np.float64).reshape(3)
    base_rot = _Rq.from_quat(np.asarray(bquat_wxyz, dtype=np.float64).reshape(4)[[1, 2, 3, 0]])

    target_qpos = {}
    for i in range(5):
        x, y = xy[i]
        pos_b = np.array([x, y, float(TOOL_REF_POS_B[i, 2])], dtype=np.float64)
        rot_ref = _Rq.from_quat(np.asarray(TOOL_REF_QUAT_XYZW_B[i], dtype=np.float64))
        rot_b = _Rq.from_euler("z", float(yaw[i])) * rot_ref      # 先乘 = 在 base 系绕 z
        world_quat_xyzw = (base_rot * rot_b).as_quat()
        target_qpos[TOOL_JOINTS[i]] = np.concatenate(
            [base_pos + base_rot.apply(pos_b), world_quat_xyzw[[3, 0, 1, 2]]])
    env.set_joint_qpos(target_qpos)
    env.mj_forward()


def _tools_pos_b(env) -> np.ndarray | None:
    """5 件工具 body 中心在 base 系的位置（校验摆放用；8 集采集器的 _query_pos_b 同源）。"""
    try:
        return np.stack(
            [np.asarray(env.query_position_body_B(j, ROBOT_BASE_BODY),
                        dtype=np.float64).reshape(3) for j in TOOL_JOINTS], axis=0)
    except Exception:
        return None


def _live_ee_b(env) -> tuple | None:
    """**实时**读右末端在 base 系的位姿 (pos, quat_xyzw)。

    为什么不能用 storage.build_state(storage.obs_callback(env))：obs 是**上一步
    env.step 时缓存的**。teleport 关节之后（还没 step）读它拿到的是旧位姿 —— 2026-09-17
    的 ik 模式就栽在这：把 OSC 目标锚到了 teleport 之前的位姿，控制器于是把手臂原样
    拽回去，teleport 白做（实测残差只从 247mm 变成 146mm）。
    这两个 query 是直接问仿真当前状态，和 probe_quat_convention.py 用的是同一条路。
    """
    from scipy.spatial.transform import Rotation as _Rq
    try:
        smat = env.query_site_pos_and_mat([EE_SITE_R])[EE_SITE_R]["xmat"]
        Rs = np.asarray(smat, dtype=np.float64).reshape(3, 3)
        spos = np.asarray(env.query_site_pos_and_quat([EE_SITE_R])[EE_SITE_R]["xpos"],
                          dtype=np.float64).reshape(3)
        bpos, bmat, _bq = env.get_body_xpos_xmat_xquat([ROBOT_BASE_BODY])
        Rb = np.asarray(bmat, dtype=np.float64).reshape(3, 3)
        bp = np.asarray(bpos, dtype=np.float64).reshape(3)
        return (Rb.T @ (spos - bp), _Rq.from_matrix(Rb.T @ Rs).as_quat())
    except Exception:
        return None


def _solve_ik_sim(env, p_t, q_t, restarts: int = 400, iters: int = 120,
                  seed: int = 0, min_margin_deg: float = 8.0) -> tuple | None:
    """在**本次运行加载的**仿真模型上现解右臂 7 关节 IK。

    为什么必须现解、不能沿用 json 里存好的关节值
    ------------------------------------------------
    仿真每次重建场景都会从 `/home/dgl/.orcagym/tmp/<UUID>.xml` **重新随机化**机器人模型：
    关节轴 / 限位 / body 位置一致，但**每节臂 body 的四元数是随机抽的**。实测
    `arm_r_link1` 的等效轴角一次是 24.62°、另一次是 21.33°（同一个 body！）。
    后果：同一组关节值在不同次运行里给出的末端位姿能差几百 mm ——
    2026-09-17 实测把 start_ref 里存的 ik_joints_r 原样写进去，直读 qpos 完全等于设定值，
    但 base 系末端停在 [-0.0385, 0.2806, 0.4581]，离目标 593.82mm
    （四条 FK 路径给出完全相同的值，所以不是量法问题）；而同一次运行里现解再写回去，
    官方接口实测 **0.00mm / 0.00°**。400 次重启约 10s。

    返回 (q7_rad, 位置残差m, 姿态残差°, 最小限位余量°)，失败返回 None。
    """
    import mujoco
    from scipy.spatial.transform import Rotation as _R
    try:
        m = env.gym._mjModel
        d = env.gym._mjData
    except AttributeError:
        return None
    names = list(E.agent_conf.r_arm["joint_names"])
    try:
        jids = [m.joint(env.joint(n)).id for n in names]
        qadr = [int(m.jnt_qposadr[j]) for j in jids]
        lo = np.array([m.jnt_range[j][0] for j in jids])
        hi = np.array([m.jnt_range[j][1] for j in jids])
        sid = m.site(EE_SITE_R).id
        bid = m.body(ROBOT_BASE_BODY).id
    except Exception:
        return None
    q0 = np.array(d.qpos).copy()
    p_t = np.asarray(p_t, dtype=np.float64).reshape(3)
    R_star = _R.from_quat(np.asarray(q_t, dtype=np.float64).reshape(4)).as_matrix()

    def _rotvec(M):
        c = float(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))
        a = float(np.arccos(c))
        if a < 1e-9:
            return np.zeros(3)
        return a / (2.0 * np.sin(a)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0],
                                                 M[1, 0] - M[0, 1]])

    # FK 用 mj_kinematics：它照样更新 xpos/xmat/site_xpos/site_xmat，但不做碰撞与动力学。
    # （mj_forward 在 qpos 被清零时会把 9 个自由物体叠到原点，碰撞检测直接段错误。）
    def _fk(q7):
        d.qpos[:] = q0
        d.qpos[qadr] = q7
        mujoco.mj_kinematics(m, d)
        Rb = np.array(d.xmat[bid]).reshape(3, 3)
        return (Rb.T @ (np.array(d.site_xpos[sid]) - np.array(d.xpos[bid])),
                Rb.T @ np.array(d.site_xmat[sid]).reshape(3, 3))

    def _err(q7):
        p, Rc = _fk(q7)
        return np.concatenate([p - p_t, _rotvec(R_star.T @ Rc)]), p, Rc

    def _margin(q7):
        return float(np.degrees(np.min(np.minimum(q7 - lo, hi - q7))))

    rng = np.random.default_rng(seed)
    h = 1e-6
    best = (np.inf, np.inf, None)
    finals = []
    for _r in range(restarts):
        q = rng.uniform(lo, hi)
        lam = 1e-2
        e, p, Rc = _err(q)
        dp = float(np.linalg.norm(p - p_t))
        da = float(np.degrees(np.linalg.norm(_rotvec(R_star.T @ Rc))))
        for _i in range(iters):
            if (dp, da) < (best[0], best[1]):
                best = (dp, da, q.copy())
            if dp < 1e-5 and da < 0.02:
                break
            J = np.zeros((6, 7))
            for i in range(7):
                qp = q.copy()
                qp[i] = q[i] + h if (q[i] + h) <= hi[i] else q[i] - h
                ep, _, _ = _err(qp)
                J[:, i] = (ep - e) / (qp[i] - q[i])
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            except np.linalg.LinAlgError:
                break
            qn = np.clip(q - dq, lo, hi)
            en, pn, Rcn = _err(qn)
            if np.linalg.norm(en) < np.linalg.norm(e):
                q, e, p, Rc = qn, en, pn, Rcn
                lam = max(lam * 0.7, 1e-6)
            else:
                lam = min(lam * 2.5, 1e3)
            dp = float(np.linalg.norm(p - p_t))
            da = float(np.degrees(np.linalg.norm(_rotvec(R_star.T @ Rc))))
        finals.append((dp, da, q.copy()))
    d.qpos[:] = q0                       # 别把探针用的草稿姿态留在仿真里
    # 先挑"精度够 + 限位余量够"的盆地里余量最大的：OSC 会把顶限位的关节一直压在边界上。
    # 2026-09-17 ik6 踩到的坑：这一批**没有一个**盆地满足余量阈值，于是退回全局最优
    # (best)，而 best 只按 (位置,姿态) 排序、不看余量 —— 选出的解 j4=3.63°，
    # j4 限位是 [0, 100.268]°，等于把关节压在限位上；OSC 顶着限位推，3 步就漂了
    # 10.44mm（ik5 选的解余量 14.92°，只漂 3.16mm）。
    # 所以退也要退得聪明：**先在所有"精度够"的盆地里取余量最大的**，只有当没有一个
    # 精度达标时才用全局最优，并且把余量报出来让调用方知道。
    ok = [(dp, da, q) for dp, da, q in finals
          if dp < 0.005 and da < 2.0 and _margin(q) > min_margin_deg]
    if ok:
        pick = max(ok, key=lambda t: _margin(t[2]))
    else:
        acc = [(dp, da, q) for dp, da, q in finals if dp < 0.005 and da < 2.0]
        if acc:
            pick = max(acc, key=lambda t: _margin(t[2]))
            print(f"    [start] ⚠ 没有盆地满足 {min_margin_deg:.1f}° 余量；"
                  f"改取精度达标者里余量最大的 {_margin(pick[2]):.2f}°"
                  f"（余量小 = 关节顶限位，OSC 会漂）", flush=True)
        elif best[2] is not None:
            pick = (best[0], best[1], best[2])
            print(f"    [start] ⚠ 没有盆地同时位置<5mm 且姿态<2°；"
                  f"取全局最优 {best[0]*1000:.2f}mm/{best[1]:.2f}°", flush=True)
        else:
            return None
    return (pick[2], float(pick[0]), float(pick[1]), _margin(pick[2]))


def _site_pos(env, name: str) -> np.ndarray | None:
    try:
        d = env.query_site_pos_and_quat([name])
        return np.asarray(d[name]["xpos"], dtype=np.float64).reshape(3)
    except Exception:
        return None


def _sites_pos(env, names: list) -> list:
    """一次批量 query 拿多个 site 的世界坐标。

    逐个调 _site_pos 是 N 次仿真调用；单回合模式每个 chunk 都要问 5 件工具，
    所以这里合并成一次。失败就退回逐个。
    """
    names = list(names)
    try:
        d = env.query_site_pos_and_quat(names)
    except Exception:
        return [_site_pos(env, n) for n in names]
    out: list = []
    for n in names:
        try:
            out.append(np.asarray(d[n]["xpos"], dtype=np.float64).reshape(3))
        except Exception:
            out.append(None)
    return out


def _body_pos(env, name: str) -> np.ndarray | None:
    try:
        xpos, _, _ = env.get_body_xpos_xmat_xquat([name])
        return np.asarray(xpos, dtype=np.float64).reshape(3)
    except Exception:
        return None


def _dist(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ---------------------------------------------------------------------------
# 进度判据：用**箱子内部范围**数"真的放进去几件"，而不是数开爪边沿
# ---------------------------------------------------------------------------
# 为什么换：cycle_idx 原来是靠"右爪闭→开边沿"推进的。这个前提是"每次开爪都真的
# 放好了一件"，实测里会两头破：
#   · 抓空/被碰开/半路掉 → 白记一次 → idx 提前跳 → 模型把没收的工具整件跳过；
#   · 模型抓着工具不放（repeat=10 那次抓着螺丝刀在箱子里僵了 128s）→ 边沿永不到来
#     → idx 卡死在 1 → 后面 4 件一件都轮不到。
# 训练数据里"开爪"和"进箱"是重合的（脚本轨迹是 放置→松开），所以按箱内容推进
# 既符合训练分布，又只在**出错的回合**里才和训练不同 —— 正是我们想要的。
#
# 标定直接用采集器自己那份函数（同一个几何、同一个 base 系），保证训练端和推理端
# 对"在箱内"的理解不分叉。
_BOX_INNER: dict | None = None      # 箱子内部范围（base 系），开局标定一次
_IN_BOX_MAX = 0                     # 已入箱件数的单调最大值（绝不回退）
_BOX_INNER_SRC = "none"
_HOLD_SINCE: dict = {}              # 工具 site -> 进入箱心 0.25m 时的 env 步号（official 判据用）
_PRE_SKIP: set = set()              # 开局就已经"在箱里/在得分"的工具 site，不计入本轮进度（见 _mark_pre_scoring）

# ---- 官方判据（2026-09-17 从 scorer_service 的 configs/tasks.yaml 抄来）------------
# task3_tool_storage 每件工具（7 分）的 success 是 **all**:
#   · site_distance: [工具 site, Group_Interactive_ToolBox_base_site]
#       score_curve: 0m→1.0 ; 0.25m→0.8571 ; **>0.2501m→0**
#   · duration: point_in_region(工具 site, center=ToolBox_base_site, radius=0.3) min_s=1.0
# 也就是说官方判的是**到箱位中心的距离**，**不是"在箱子内部"**；而且还要在 0.3m 内
# **连续待够 1.0 秒**才作数（被搬过去又拎走的不算）。
# 这张表比采集器的箱底内面矩形**宽**：真的放进箱底的工具 d≈0.10~0.17m，两边都过；
# 但"停在箱子边沿/箱口上方"这种，官方给分、箱内矩形不给。既然评分用的是前者，
# 进度信号就用前者 —— 让 idx 恰好在"这件工具开始得分"时推进。
OFFICIAL_SCORE_DIST = 0.25          # site_distance 给分上限（超过 0.2501 记 0 分）
OFFICIAL_HOLD_R = 0.30              # duration 门槛：箱心 0.3m 内
OFFICIAL_HOLD_S = 1.0               # 且要连续 1.0 秒


# ---------------------------------------------------------------------------
# 机器模型一致性自检：**只回答"模型文件是不是作者那份"**，不回答"视觉能不能用"
# ---------------------------------------------------------------------------
# ⚠ 2026-09-20 更正（看画面之前，这个自检一直被当成"视觉 OOD"的证据，是**过度推断**）：
#   当天 `task3_frames_idxR/` 里的头相机和腕相机画面**我逐张看过，都是正的、正常的**
#   （箱盖、绿桌、5 件工具一字排开、地平线水平），而当时的自检报
#   head_link1 偏 89.98°、arm_r_link1 偏 115.14°。**88 度的数字配一张正的画面**，
#   说明这个数根本不是相机的朝向。中间 link 差 90° 完全可能被下游 link 补回来。
#   所以那两行"视觉等于没用/画面不可与训练帧比对"是**错的**，已改掉。
#   顺带查出来源：老实现读的是 `_mjData.xmat` = **当前位姿**下的世界朝向，
#   而参照值是**静息**值 ⇒ 关节一动就报"模型被换了"，一天到晚在报假警。
#   已改成 `_rest_world_rot()`（沿父链组合 `mjModel.body_quat`，与位姿无关）。
#   要判视觉，唯一的判据是**把画面调出来看**，或与训练帧 0 做相关。
#
# 下面这段 09-17 的测量是**真的**（相机确实会转、相关确实会掉），保留备查：
# 2026-09-17 查清的事实（见 README 二·九 与 memory/orca-sim-robot-model-differs-from-xml）：
# 仿真每次加载场景都会重写机器人的**每节 link 的静息变换**，同一台 server 里隔几分钟
# 变一次。头相机和腕相机都挂在那些 link 上：
#     camera_head_body1  ← head_link2 ← head_link1 ← body_link2
#     camera_right_body1 ← arm_r_end_link ← … ← arm_r_link1 ← body_link2
#     （body_link2 与 body_link1 之间**没有关节**，是焊死的，所以没有腰关节）
# 于是 link 一转，**相机朝向就跟着转，而 22 维 state 里没有任何一维能反映它**
# （头关节 idx11/idx12 也不在 state 里，而且一直是 0）。
#
# 实测铁证（同一个训练帧 0 作参照，头相机相关系数随时间单调掉下去）：
#   14:14 ik  +0.758   14:19 ik2 +0.755   14:31 ik3 +0.576   14:38 ik4 +0.298
#   14:49 ik5 +0.089   14:56 ik6 +0.076   14:58 ik7 +0.005   15:01 ik8 +0.133
#   同期 head_link1 的 euler z：≈0 → −5.50(14:45) → −30.43(15:00)
# 就是说：**头相机被转走了 30°，画面就从"同分布"变成"完全不相干"**。
# 而 arm_r_link1 的 z 今天在 0/±180/127.5/123.2/−42.0/−49.2/−58.2/−42.7/96.0/
# 119.1/107.1/102.0 之间跳 —— 腕相机挂在它下游，所以**今天没有任何一次运行的腕
# 相机画面接近过训练**（最好的也只有 +0.303）。
# 这也解释了"为什么把右臂对齐到 0.00mm 也没用"：对齐的是 EE，管不了相机。
# ↑ 注意：上面这段的结论是"**图像相关系数**会掉"，证据是相关系数本身。
#   它不是"link 姿态差多少度"推出来的 —— 两者别混。
#
# 下面是**只报警、不改模型**的自检：把当前**加载文件**里的 link 静息朝向和
# "作者值"比一下，不同就报出来。作者值取 08:35 那次加载（euler 是整齐的整数）。
# ⚠ 作者值本身可能是**过期**的：08:35 之后没人再记录过 roll，只记录过 z。
#   所以报出来时只能说"与 08:35 那份不同"，**不能**说"这是错的/相机废了"。
_MODEL_REF = {
    # body 名: 干净作者 euler(xyz, 度)
    "g1_omnipicker_head_link1": (90.0, 0.0, 0.0),
    "g1_omnipicker_body_link2": (-90.0, 0.0, 0.0),
    "g1_omnipicker_arm_r_link1": (180.0, 0.0, 180.0),
}


def _rest_world_rot(mjm, bid: int):
    """body 的**静息**世界朝向 = 所有关节都在 0 位时的朝向（沿父链组合 body_quat）。

    为什么要自己算、不用 `_mjData.xmat`（2026-09-20 修的 bug）：
    `xmat` 是**当前位姿**下的世界朝向，关节一动它就变。拿它去比"作者静息值"，
    等于**手臂一抬起来就报"模型被换了"** —— 这是假报警，而且它一天到晚在报。
    实测坐实：`task3_frames_idxR/` 里今天的头/腕相机画面都是**正的**（我逐张看过），
    可当时的自检报 head_link1 偏 89.98°、arm_r_link1 偏 115.14°。
    也就是说那两行字**跟"相机能不能用"根本没有关系**。
    `mjModel.body_quat` 是相对父节点的**静息**朝向，与位姿无关，这才是"模型身份"。
    """
    from scipy.spatial.transform import Rotation as _Rm
    chain, b = [], int(bid)
    while b > 0:                          # body 0 = world，parentid 也是 0
        chain.append(b)
        b = int(mjm.body_parentid[b])
    rot = _Rm.identity()
    for b in reversed(chain):             # 自根向下组合
        q = np.asarray(mjm.body_quat[b], dtype=np.float64)     # MuJoCo: w x y z
        rot = rot * _Rm.from_quat([q[1], q[2], q[3], q[0]])
    return rot


def _check_model_consistency(env, warn_deg: float = 2.0) -> dict:
    """比"当前**加载文件**里的 link 静息朝向"与"干净作者值"，返回 {body: 偏差角(度)}。

    **这个数只说明"模型文件是不是作者那份"，不说明相机能不能用**（见 _rest_world_rot）。
    想知道视觉有没有问题，唯一的判据是**把画面调出来看**，或者拿它跟训练帧 0 做相关。

    比的是**旋转矩阵之间的测地角**，不是 euler 字符串 —— `(180,0,180)` 和
    `(180,0,-180)` 是同一个旋转，按字符串比会误报 360°。
    ⚠ 约定：MuJoCo 的 `euler` 是**内旋**（绕移动轴依次转，默认 eulerseq="xyz"），
    对应 scipy 的**大写** `"XYZ"`；scipy 小写 `"xyz"` 是外旋，两者只在多轴同时
    非零时不同 —— arm_r_link1 的 (180,0,180) 正好是多轴，用错会白报 70°。
    """
    from scipy.spatial.transform import Rotation as _Rm
    dev: dict = {}
    for nm, eul in _MODEL_REF.items():
        try:
            bid = env.gym._mjModel.body(nm).id
            cur = _rest_world_rot(env.gym._mjModel, bid)
            ref = _Rm.from_euler("XYZ", np.asarray(eul, dtype=np.float64),
                                 degrees=True)
            dev[nm] = float(np.degrees((ref.inv() * cur).magnitude()))
        except Exception as exc:
            print(f"[model] ⚠ 读不到 {nm} 的静息朝向（{type(exc).__name__}: {exc}）",
                  flush=True)
    if not dev:
        return dev
    bad = {k: v for k, v in dev.items() if v > warn_deg}
    print("[model] 机器模型一致性自检（静息朝向 vs 作者值；与当前位姿无关）：", flush=True)
    for k, v in dev.items():
        print(f"    {k:38s} 偏差 {v:7.2f}°{'   ← 偏了' if v > warn_deg else ''}",
              flush=True)
    if bad:
        worst = max(bad.values())
        print(f"[model] ⚠ 机器人模型文件与作者那份**不同**（最大偏差 {worst:.1f}°）。"
              f"注意这个数**只说明模型文件不同**，不能推出「视觉没用」：中间 link 差 90°"
              f"完全可能被下游 link 补回来（今天就是这种情况 —— 画面是正的）。"
              f"要判视觉，**看画面**，别信这个数。", flush=True)
    else:
        print("[model] ✓ link 静息朝向与作者值一致。", flush=True)
    return dev


def _calibrate_box_inner(env) -> dict | None:
    """量出箱子内部范围。优先复用采集器的标定函数。"""
    global _BOX_INNER, _BOX_INNER_SRC
    try:
        _dc = os.path.abspath(os.path.join(base_dir, "..", "..", "dataCollection", "g1_omnipicker"))
        if _dc not in sys.path:
            sys.path.insert(0, _dc)
        import g1_omnipicker_collection_scripted_tool_lerobot as _T
        # 注意：必须传**全名** ROBOT_BASE_BODY。"g1_omnipicker_body_link1" 是全名，
        # 而 agent_conf.base_body 是 "body_link1"（短名），直接传会 KeyError。
        info = _T._calibrate_box_bottom_inner(env, ROBOT_BASE_BODY)
        _BOX_INNER = {
            "xy_min": np.asarray(info["xy_min"], dtype=np.float64),
            "xy_max": np.asarray(info["xy_max"], dtype=np.float64),
            "z_surface": float(info["z_surface"]),
        }
        _BOX_INNER_SRC = "采集器标定(箱底内面矩形)"
        print(f"[progress] 箱子内部范围标定完成 [{_BOX_INNER_SRC}]: "
              f"xy {_BOX_INNER['xy_min'].round(3).tolist()}~{_BOX_INNER['xy_max'].round(3).tolist()}"
              f"  z_surface={_BOX_INNER['z_surface']:.3f}", flush=True)
        return _BOX_INNER
    except Exception as exc:
        _BOX_INNER = None
        _BOX_INNER_SRC = f"失败({type(exc).__name__}: {exc})"
        print(f"[progress] ⚠ 箱子内部标定失败：{_BOX_INNER_SRC}；"
              f"退回箱位半径 {PLACED_DIST:.2f}m 判据", flush=True)
        return None


def _count_in_box(env, tool_sites: list, box0) -> int | None:
    """数有几件工具真的在箱子内部；返回单调最大值（None = 判据不可用）。

    base 系里比：xy 落在箱底内面矩形内，z 在 [底面上表面-6cm, +15cm]。
    z 上限放宽是因为工具是**被夹爪放进去**的：松手前后、落稳之前都在箱内高度
    范围内，判太紧会在刚松手那几帧漏记，而漏记的后果就是 idx 不推进。
    """
    global _IN_BOX_MAX
    if _BOX_INNER is None:
        # 退回老判据（箱位半径），保证标定失败时行为不低于改动前
        if box0 is None:
            return None
        cur = _sites_pos(env, tool_sites)
        n = sum(1 for s, p in zip(tool_sites, cur)
                if s not in _PRE_SKIP and _dist(p, box0) <= PLACED_DIST)
        _IN_BOX_MAX = max(_IN_BOX_MAX, n)
        return _IN_BOX_MAX
    try:
        # 全名，否则 KeyError（query_site_pos_and_quat_B 不做短名解析）→ 被下面吞掉变哑巴
        d = env.query_site_pos_and_quat_B(list(tool_sites), [ROBOT_BASE_BODY])
    except Exception:
        return _IN_BOX_MAX if _IN_BOX_MAX else None
    lo = _BOX_INNER["xy_min"]
    hi = _BOX_INNER["xy_max"]
    zs = _BOX_INNER["z_surface"]
    n = 0
    for s in tool_sites:
        if s in _PRE_SKIP:
            continue
        try:
            p = np.asarray(d[s]["xpos"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        if (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]
                and (zs - 0.06) <= p[2] <= (zs + 0.15)):
            n += 1
    _IN_BOX_MAX = max(_IN_BOX_MAX, n)
    return _IN_BOX_MAX


def _count_scoring(env, tool_sites: list, box0, cur_step: int) -> int | None:
    """数有几件工具**正在得分**（官方判据）；返回单调最大值。

    官方判据是两件一起满足：`site_distance ≤ 0.25m` **且** 在**箱心 0.3m 内连续待够
    1.0 秒**。两个都照做：
      · 距离用 OFFICIAL_SCORE_DIST(0.25)；
      · 时长用 env 步数折算（1 env 步 = 5ms 仿真，所以 1.0s = 200 步）。
    为什么要那个时长门槛：模型**提着**工具从箱子上方经过时也会短暂进入 0.25m，
    只按瞬时距离算就会在还没松手时就推进 idx —— 模型于是收到"下一轮"的 state
    却还攥着上一件，正是我们要避免的错位。
    """
    global _IN_BOX_MAX
    if box0 is None:
        return None
    cur = _sites_pos(env, tool_sites)
    need = int(round(OFFICIAL_HOLD_S / 0.005))     # 200 步 = 1.0s 仿真
    n = 0
    for site, p in zip(tool_sites, cur):
        if site in _PRE_SKIP:
            # 开局就在箱心 0.25m 内的那件（本场景：手电筒，0.084m）不算"这一轮搬进去的"。
            # 不排除的话开局 1.0s 后 n 就变 1、cycle_idx 从 1 跳到 2，模型收到"下一轮"
            # 的 state 却一件都还没搬 —— 与训练帧 0 的 cycle_idx=1 直接冲突。
            continue
        if _dist(p, box0) <= OFFICIAL_SCORE_DIST:
            t0 = _HOLD_SINCE.get(site)
            if t0 is None:
                _HOLD_SINCE[site] = cur_step
                t0 = cur_step
            if (cur_step - t0) >= need:
                n += 1
        else:
            _HOLD_SINCE.pop(site, None)
    _IN_BOX_MAX = max(_IN_BOX_MAX, n)
    return _IN_BOX_MAX


def _mark_pre_scoring(env, tool_sites: list, box0, gate: str) -> set:
    """开局就把"已经在箱里"的工具记下来，本轮进度不再数它们。

    为什么需要：本评测场景**开局就把手电筒摆在箱边 0.084m 处**（其余 4 件在
    0.355/0.409/0.474/0.799m）。官方判据只看"离箱心 ≤0.25m 且持续 1.0s"，
    于是开局 1.0s 后计数就从 0 变 1、cycle_idx 从 1 跳到 2 ——
    机器人什么都没干，模型却收到"已经搬完一件、去抓第二件（螺丝刀）"的 state：
    goal_y 立刻从 -0.0845(扳手) 变成 -0.2158(螺丝刀)。训练帧 0 是 cycle_idx=1、
    目标扳手，这正是要避免的错位。

    判据只取**开局瞬时**距离（不需要 1.0s 门槛）：开局没有"到达事件"可言，
    那件工具本来就是摆在那儿的。
    """
    if box0 is None:
        return set()
    try:
        if gate == "box":
            cur = _sites_pos(env, tool_sites)
            d = {s: _dist(p, box0) for s, p in zip(tool_sites, cur)}
            thresh = PLACED_DIST
        else:
            d = {s: _dist(p, box0) for s, p in zip(tool_sites, _sites_pos(env, tool_sites))}
            thresh = OFFICIAL_SCORE_DIST
    except Exception as exc:
        print(f"[progress] 开局基准判定失败（{type(exc).__name__}: {exc}）；"
              f"按全部都要搬处理", flush=True)
        return set()
    pre = {s for s, v in d.items() if v <= thresh}
    if pre:
        names = ", ".join(f"{s}({d[s]:.3f}m)" for s in sorted(pre))
        print(f"[progress] 开局已达标，不计入本轮进度：{names}"
              f"（否则 cycle_idx 会开局 +{len(pre)}）", flush=True)
    return pre


def _grip_open(device) -> bool:
    """按右夹爪电机目标判断是否张开。"""
    ctrl = np.asarray(device.r_grip_ctrl, dtype=np.float64).reshape(-1)
    return float(ctrl[0]) < GRIP_OPEN_MOTOR


def _dump_frames(cameras: dict, tag: str, out_dir: str) -> None:
    """把当前各相机帧存成 PNG，供事后核对策略看到什么画面。"""
    if not cameras or not out_dir:
        return
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    for name, cam in cameras.items():
        try:
            f, _ = cam.get_frame(format="rgb24")
            if f is not None and f.size:
                p = os.path.join(out_dir, f"{tag}_{name}.png")
                cv2.imwrite(p, cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Task3 工具收纳评分接入")
    ap.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml")
    ap.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--team-token", default=None)
    ap.add_argument("--robot-id", default="g1_omnipicker")
    ap.add_argument("--dry-run", action="store_true", help="只读探测场景+测速，不评分不动工具")
    ap.add_argument("--no-score", action="store_true", help="跑动作但不评分（演练用）")
    # 【2026-09-17 更新：下面这段老注释里"25s 阶段预算"的理由已经作废】
    # 老理由是"单阶段预算 (165-已用)/(剩余+1)-3 ≈ 25 s 撑不住 45.7 s 的工具"，但那是
    # **--phased 5 阶段模式**才有的预算；默认的单回合连续模式没有分阶段预算，所以
    # "改成 10 就 0 件入箱"不再成立。今天两种都实测过（都 --no-score、都 pos 对齐）：
    #   repeat=1  kp=150：2/5 入箱；cycle_idx 1→2→3 正常推进；165s 只换到 31 s 仿真。
    #   repeat=10 kp=220：1/5；12890 步 = **64.5 s 仿真（2 倍）**；手臂真的抓起了螺丝刀
    #                     并搬了 445 mm 进箱子 —— 物理上明显更像训练，但**闭合后再没张开**，
    #                     128 s 里 0 条张开指令 → 没有 release → cycle_idx 卡死在 1。
    # 所以 10/220 是"仿真时间翻倍、动作更真"的正确方向，但它把真正的短板暴露出来了：
    # 模型不松手。松手是 cycle_idx 前进的唯一途径，而不松手就不会有第 2、3 件。
    # 在松手问题解决之前，默认仍取实测分更高的 1/150；要跑对照就显式传 10/220。
    ap.add_argument("--action_repeat", type=int, default=1,
                    help="每个动作重复几个 env 步。1=与训练时间轴不符但实测分更高(2/5)；"
                         "10=与训练完全一致、同样墙钟能换到 2 倍仿真时间，但模型不松手(1/5)。")
    ap.add_argument("--kp", type=float, default=150.0,
                    help="右臂 OSC 位置刚度。训练采集用的是 220。")
    ap.add_argument("--grasp_integral", action="store_true")
    ap.add_argument("--settle_steps", type=int, default=180,
                    help="开局复刻采集器的物理沉降步数（采集器默认就是 180）。"
                         "采集器的初始化顺序是 reset → set_default_joint_values → "
                         "用全 0 的 init ctrl 沉降 180 个 env 步 → 再 set_default + mj_forward，"
                         "然后才开始录帧。④ 原来只有 8 步，起始状态和采集时不是同一条轨迹上来的。"
                         "设 0 关闭（退回老行为）。")
    ap.add_argument("--plant_layout", type=str, default="nominal",
                    choices=("nominal", "jitter", "off"),
                    help="开局把 5 件工具种回训练用过的标称槽位（采集器每集都这么做，"
                         "而 reset()/update_scene() 不复位 free-joint 工具）。"
                         "nominal=纯标称（默认，确定性）；jitter=按采集器默认幅度随机；"
                         "off=沿用关卡自带摆放（旧行为，手电筒会在箱边、开局即得分）。")
    ap.add_argument("--plant_seed", type=int, default=0)
    ap.add_argument("--progress_gate", type=str, default="grip",
                    choices=("grip", "box", "official"),
                    help="cycle_idx 的推进源。**默认 grip**，理由是可实测的，不是偏好："
                         "训练数据里 idx 的跳变帧恰好是右爪『闭→开』边沿的**下一帧**"
                         "（2026-09-20 实测：8/8 集、5/5 个边界，差值恒为 +1）。"
                         "storage 的 grip 分支（lerobot_data_storage.py:1186）"
                         "与采集端的 build_d22_cycle_dataset.py 逐字对齐，"
                         "所以只有 grip 能保证『模型看到的 idx 演化』与训练同分布。"
                         "box=用采集器的箱底内面矩形数真实入箱件数。"
                         "official=用官方评分器判据（工具到箱位中心 ≤0.25m 且持续 1.0s）"
                         "数正在得分的件数。"
                         "⚠ 后两者都是**事件判据**，与训练的『松手』判据不同："
                         "2026-09-20 那轮就是被 official 坑的 —— 模型空捏着什么都没放，"
                         "却因为把工具推进了 0.20~0.25m 那条带（PLACED_DIST=0.20 < "
                         "OFFICIAL_SCORE_DIST=0.25）在 t=28.1s 被推了一次 idx，"
                         "模型收到训练里不可能出现的 state。"
                         "要复现官方给分口径请显式传 --progress_gate official。")
    ap.add_argument("--progress_baseline", type=str, default="initial",
                    choices=("initial", "none"),
                    help="cycle_idx 的计数基准。initial（默认）=**开局就在箱里/已在得分的工具"
                         "不计入本轮进度**：本评测场景开局把手电筒摆在箱边 0.084m 处，"
                         "不扣除的话开局 1.0s 后计数就变 1、cycle_idx 从 1 跳到 2，"
                         "模型收到「下一轮」的 state（goal_y 从扳手 -0.0845 变螺丝刀 -0.2158）"
                         "却一件都还没搬 —— 与训练帧 0 的 cycle_idx=1 冲突。"
                         "none=老行为（含开局那件）。")
    ap.add_argument("--camera_warmup_steps", type=int, default=10)
    ap.add_argument("--smoke", type=int, default=0, help="连相机+策略后跑 N 步推理即退出(验证管线，不评分)")
    ap.add_argument("--per_phase_max_s", type=float, default=999.0, help="单工具最长墙钟秒(0=自动均分)")
    ap.add_argument("--only_phases", type=str, default="", help="逗号分隔的 step_id 子集(如 pick_wrench)，默认全部")
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--jsonl", type=str, default="")
    ap.add_argument("--frames_dir", type=str, default="/home/dgl/lxc/.runlogs/task3_frames")
    ap.add_argument(
        "--dump_chunks",
        type=str,
        default="",
        help="把每次推理的原始输出转储成 JSONL（默认关闭，不影响任何现有行为）。"
             "每条 = 一次推理：19 维 state + horizon×18 的动作块 + 当时的 ee/base/工具/盒子世界坐标。"
             "动作块的第 7:10 维就是策略想要去的右臂末端位置(base 系)，与 --tool_refs 里的参照点同坐标系，"
             "所以把它和 6 个参照点比 dist、取 argmin 就能读出'它这一步在瞄哪个'。",
    )
    ap.add_argument(
        "--tool_refs",
        type=str,
        default=os.path.join(base_dir, "tool_refs_d22.json"),
        help="参照点 json，**必须与服务端那一档配对**：d19 模型 → tool_refs_d19goal.json"
             "(默认, goals = 5 工具 + 1 盒子共 6 点, state 19 维)；d22 模型 → tool_refs_d22.json"
             "(多声明 new_dims, 额外启用第 20~22 维 goal_y/cycle_dist/cycle_idx, state 22 维)。"
             "配错不会静默降级 —— 连上策略服务后会拿它握手时的 state_dim 硬校验并报错。"
             "传空串 '' 则退回 18 维老接口(v1 的 tool_refs_d19.json 仍可用但已过时)",
    )
    ap.add_argument("--phased", action="store_true",
                    help="退回旧的 5 阶段写法（每阶段一个裸工具名 prompt + 独立时间预算 + 只 reset_cycle 一次）。"
                         "**默认是单回合连续跑**，保留这个开关只为复现旧行为做对照。")
    ap.add_argument("--episode_prompt", type=str, default=EPISODE_PROMPT,
                    help="单回合模式的 prompt。默认取训练语料里的一句整场景话"
                         "（训练只有 3 句、全不含工具名，29/40/31 集轮换）。")
    ap.add_argument("--stop_cycle_idx", type=int, default=6,
                    help="停止条件之一：state 第22维 cycle_idx 达到此值即停。训练里第 5 次释放后 "
                         "idx=6 且 demo 静止 6.75s —— 那就是模型学到的『干完了』。"
                         "pi0.5 的 18 维动作里没有任何 done 位，idx 是它唯一的停止信号。")
    ap.add_argument("--stop_settle_s", type=float, default=5.0,
                    help="idx 达标后再跑这么多秒让最后一件落稳（训练在那之后还静止 6.75s）")
    ap.add_argument("--soft_deadline_s", type=float, default=None,
                    help="挂钟软时限（秒）。**不给**时自动选：--no-score 演练 → 不限时"
                         "（只按『五件全部入箱』或手动 Ctrl-C 停）；评分 → 165s"
                         "（本地 scorer 硬限 180s，超了 attempt 会被掐掉）。"
                         "0 = 强制不限时。>0 = 用这个秒数。"
                         "2026-09-20 加：演练里 165s 只买到 ~32s 仿真，五件要 ~228s，"
                         "所以那条时限一定先于真终止条件触发，永远等不到『收完了』。")
    ap.add_argument("--start_ref", type=str,
                    default=os.path.join(base_dir, "start_ref_d22.json"),
                    help="训练帧0 的 state json（起始对齐用）。传空串关闭。")
    ap.add_argument("--no_start_servo", action="store_true", help="不做起始对齐（保留旧起始位姿）")
    ap.add_argument("--start_align", type=str, default="pos", choices=("pos", "pose", "ik"),
                    help="起始对齐方式。"
                         "pos=笛卡尔伺服只较位置，实测 16.58mm/**姿态差 141.59°**；"
                         "pose=连姿态一起较，实测 50.6mm/17.88°（j4、j6 顶限位）；"
                         "ik=**不走伺服**，直接把右臂 7 个关节 teleport 到离线全关节 IK 解"
                         "（start_ref_d22.json 的 ik_joints_r），实测 9.08mm/**0.86°**。"
                         "ik 在位置和姿态上同时优于前两者 —— 姿态尤其关键，腕相机 cam_wrist_r "
                         "装在腕上，姿态差 140° 等于视向整个转掉。"
                         "（旧注释里 'pos 2/5 优于 pose 1/5' 是姿态不可达结论下的判断，"
                         "那个结论已被全关节搜索推翻。）")
    ap.add_argument("--start_servo_max_s", type=float, default=10.0, help="起始对齐最长墙钟秒")
    ap.add_argument("--ik_restarts", type=int, default=400,
                    help="ik 模式现场解 IK 的随机重启次数（每次 <=120 步 LM）。"
                         # ⚠ argparse 会对 help 做 % 插值，裸的 100% 会让整个 --help
                         #   抛 ValueError: unsupported format character。要写 100%%。
                         "400 次约 10s，命中率 100%%（见 _solve_ik_sim 注释）。")
    ap.add_argument("--ik_seed", type=int, default=0, help="ik 模式现场解 IK 的随机种子")
    ap.add_argument("--ik_min_margin_deg", type=float, default=8.0,
                    help="ik 模式接受的解必须离关节限位至少这么多度（OSC 会顶限位）")
    args = ap.parse_args()
    args.only_phases = {s.strip() for s in args.only_phases.split(",") if s.strip()}

    # ---- 挂钟软时限：演练不限时，评分才锁 165s（见 _resolve_soft_deadline）----
    global SOFT_DEADLINE_S
    SOFT_DEADLINE_S = _resolve_soft_deadline(args.no_score, args.soft_deadline_s)
    if np.isfinite(SOFT_DEADLINE_S):
        _dl_why = "评分路径（本地 scorer 硬限 180s，预留 finish+上传）"
    else:
        _dl_why = "演练 --no-score → 不限时，只按『五件全部入箱』或手动 Ctrl-C 停"
    print(f"[time] 挂钟软时限 = {_deadline_text(SOFT_DEADLINE_S)}   {_dl_why}", flush=True)

    # ---- summary 落盘路径：--dump_chunks 给了就顺带推导一份，别再多记一个开关 ----
    # 2026-09-20 栽过：命令里只写了 --dump_chunks 没写 --jsonl，于是 4.9MB 的原始转储在、
    # **总结那份一行都没有**（`--jsonl` 默认是空串 = 不写）。不限时模式下人工 Ctrl-C 是常态，
    # 而"半截记录"也走 args.jsonl ⇒ 那次停机的 n_in_box / cycle_trace 全丢了。
    # 现在：--jsonl 没给但 --dump_chunks 给了 ⇒ 从转储名推导 `<dump 去后缀>_summary.json`。
    if not args.jsonl and args.dump_chunks:
        _dp = os.path.abspath(args.dump_chunks)
        args.jsonl = (os.path.splitext(_dp)[0] + "_summary.json") if _dp.endswith(".jsonl") \
            else (_dp + "_summary.json")
        print(f"[jsonl] 你没给 --jsonl，从 --dump_chunks 推导 summary 路径 -> {args.jsonl}",
              flush=True)
    elif not args.jsonl:
        print("[jsonl] ⚠ --jsonl 和 --dump_chunks 都没给：这一轮**不会留任何记录**，"
              "停机后无法复盘。想留记录加 --dump_chunks <path>.jsonl", flush=True)

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        from yaml import Loader, load
        config = load(f, Loader=Loader)

    tool_refs = None
    ref_name = "tool_dist"
    cycle_refs = None
    if args.tool_refs:
        with open(args.tool_refs, "r", encoding="utf-8") as f:
            _tr = json.load(f)
        # goals(v2, 6 点) 优先于 tool_refs(v1, 5 点)。这个 json 两个键都写了，
        # 只读 tool_refs 会静默退化成 v1 —— 那训练出来的模型收到的是错的输入，
        # 所以这里必须优先 goals，并把点数打出来供核对。
        if "goals" in _tr:
            tool_refs = np.asarray(_tr["goals"], dtype=np.float32).reshape(-1, 3)
            ref_name = "goal_dist"
            print(f"[setup] d19 v2 已载入 {len(tool_refs)} 个参照点(5 工具 + 1 盒子) -> state = 19 维, "
                  f"第19维名为 goal_dist", flush=True)
        else:
            tool_refs = np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
            ref_name = "tool_dist"
            print(f"[setup] d19 v1 已载入 {len(tool_refs)} 个工具参照点 -> state = 19 维, "
                  f"第19维名为 tool_dist（注意：这需要配 v1 训练的模型）", flush=True)
        # json 自己声明 new_dims 才开第20~22维。用声明而不是命令行开关：忘了加开关
        # 是静默错误（state 少 3 维，模型看到没见过的 prompt），json 里写没写至少是显式的。
        if _tr.get("new_dims"):
            cycle_refs = np.asarray(_tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
            print(f"[setup] 该 json 声明 new_dims={_tr['new_dims']} -> state = 22 维 "
                  f"(轮次目标取自有序的 {len(cycle_refs)} 个工具抓取点)", flush=True)
    else:
        print("[setup] 未传 --tool_refs -> state = 18 维（第19维会被服务端补零成恒定 0.18m）", flush=True)

    scene_manager = E.SceneManager(args.orcagym_addr, config=config)
    storage = E.G1OmniPickerLeRobotStorage(
        dataset_path="/tmp/_task3_scratch", tool_refs=tool_refs, ref_name=ref_name,
        cycle_refs=cycle_refs,
    )
    # 单次 attempt 单进程，计数本来就从 0 起；显式调一次是为了让"这里有状态"可见，
    # 将来改成多轮 attempt 时也不会忘记清零。
    storage.reset_cycle()
    manager = E.DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=E.ENTRY_POINT,
        default_joint_values={},
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([E.agent_conf.positions_group])
    manager.set_task(E.EmptyTask(env))
    manager.mode = E.DataCollectionManager.DataCollectionMode.INFERENCE

    # ---- 规范化初始场景 ----
    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        raise RuntimeError("场景更新失败")
    env.set_default_joint_values(E.build_default_joint_values())
    env.mj_forward()

    # ---- 种回 5 件工具的标称槽位（与采集器同序：plant → 后面还会 settle 180 步）----
    # reset()/update_scene() 都不复位 free-joint 工具，不种的话会沿用关卡自带摆放
    # （实测手电筒在箱边 0.084m，开局即得分）。详见 TOOL_REF_POS_B 处的注释。
    if args.plant_layout != "off":
        _t0 = time.time()
        if args.plant_layout == "jitter":
            _rng = np.random.default_rng(args.plant_seed)
            _xy, _yaw = _sample_tool_layout(_rng, **TOOL_JITTER_TRAIN)
            _tag = f"抖动 {TOOL_JITTER_TRAIN}"
        else:
            _xy = {i: (float(TOOL_REF_POS_B[i, 0]), float(TOOL_REF_POS_B[i, 1]))
                   for i in range(5)}
            _yaw = {i: 0.0 for i in range(5)}
            _tag = "纯标称"
        _plant_tools(env, _xy, _yaw)
        _pb = _tools_pos_b(env)
        _dev = (float(np.abs(_pb - TOOL_REF_POS_B).max() * 1000)
                if _pb is not None else float("nan"))
        print(f"[plant] 已种回 5 件工具（{_tag}，seed={args.plant_seed}，"
              f"{time.time()-_t0:.2f}s）最大偏离标称 {_dev:.1f}mm", flush=True)
        if _pb is not None:
            for _i in range(5):
                print(f"[plant]   {TOOL_LABELS[_i]:<10} base 系 "
                      f"{np.round(_pb[_i], 4).tolist()}  "
                      f"标称 {np.round(TOOL_REF_POS_B[_i], 4).tolist()}", flush=True)

    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    for c in manager.controllers:
        c.reset()
    env.render()
    time.sleep(0.05)

    # 机器模型自检：相机挂在这些 link 上，link 一变相机朝向就变（见 _check_model_consistency）
    try:
        _MODEL_DEV = _check_model_consistency(env)
    except Exception as exc:
        print(f"[model] 自检本身出错（{type(exc).__name__}: {exc}），跳过", flush=True)

    # ---- 只读场景探测 ----
    print("\n[probe] === 当前 ORCA 场景探测 ===", flush=True)
    box = _site_pos(env, TOOLBOX_SITE)
    base = _body_pos(env, ROBOT_BASE_BODY)
    print(f"[probe] toolbox site {TOOLBOX_SITE}: {box}", flush=True)
    print(f"[probe] robot base  {ROBOT_BASE_BODY}: {base}", flush=True)
    reach_d = _dist(base, box)
    print(f"[probe] base→toolbox = {reach_d:.3f} m  (reach 满分带 ~0.4~1.0, 峰值 0.65)", flush=True)

    tool_init: dict = {}
    missing: list[str] = []
    for ph in SCORE_PHASES:
        p = _site_pos(env, ph["site"])
        if p is None:
            missing.append(ph["site"])
            print(f"[probe]  !! 缺少 site: {ph['site']}  step={ph['step_id']} prompt={ph['prompt']}", flush=True)
        else:
            tool_init[ph["site"]] = p.tolist()
            print(f"[probe]  {ph['prompt']:<10} dist_toolbox={_dist(p, box):6.3f}  @ {p}", flush=True)
    if missing:
        print(f"\n[probe] 缺少 {len(missing)} 个评分 site —— 当前布局不是 Task3 官方场景，无法评分！", flush=True)
        print("[probe] 请在 OrcaLab 加载含 Group_Interactive_*_task_*_site 的官方 Task3 布局。", flush=True)
        env.close(); return
    if base is None or box is None:
        print("\n[probe] 找不到机器人 base 或 toolbox site。", flush=True)
        env.close(); return
    if reach_d > 1.2 or reach_d < 0.3:
        print(f"[probe] 警告: base 距 toolbox {reach_d:.2f}m，reach 分可能偏低。", flush=True)

    # ---- 建立末端控制器与设备 ----
    l_arm = E.create_arm(env, E.agent_conf.l_arm)
    r_arm = E.create_arm(env, E.agent_conf.r_arm)
    kp = float(np.clip(args.kp, 1.0, 300.0))
    for _a in (l_arm, r_arm):
        _a.controller.kp = np.ones(6, dtype=np.float64) * kp
        _a.controller.kd = 2.0 * np.sqrt(_a.controller.kp)
    l_grip = E.create_gripper(env, E.agent_conf.gripper_l)
    r_grip = E.create_gripper(env, E.agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    _init_state = storage.build_state(storage.obs_callback(env))
    _init_apply = E.action_dict_for_apply(E.build_initial_action_from_state(_init_state))
    _init_apply["l_grip_ctrl"] = np.array([-0.8561, -0.8561], dtype=np.float32)
    device = E.EEFDevice(
        l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
        lock_left_arm=True, grasp_integral=args.grasp_integral, **_init_apply,
    )
    manager.set_device(device)
    device.set_left_hold(_init_apply["l_pos_b"], _init_apply["l_quat_b"], _init_apply.get("l_grip_ctrl"))
    for _ in range(8):
        act = manager.run_controllers()
        env.step(act)
        env.render()

    # ---- 复刻采集器的物理沉降（--settle_steps）----
    # 2026-09-17 查采集器 g1_omnipicker_collect_clear5_lerobot.py 得到的事实：
    # 它开机是 reset → update_scene → set_default_joint_values →
    # `_settle(env, manager, 180)`（set_init_ctrl + 用全 0 的 init ctrl 跑 180 个 env 步）
    # → set_default_joint_values + mj_forward，**然后才开始录帧**。
    # 而右臂的 `motors_init_ctrl` 是全 0、不是 conf neutral。所以训练数据是从
    # "全 0 目标沉降后的稳态"开始走的，④ 原来只跑 8 步控制步，起点不在同一条轨迹上。
    # 实测这个沉降把手臂停在 (0.728,-0.278,0.157)（低、朝前伸）—— 这就是现场看到的
    # "右臂先垂下去"。跑够步数后再对齐，起始状态才和采集时同源。
    if args.settle_steps > 0 and not args.dry_run:
        _ts0 = time.time()
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        for _ in range(int(args.settle_steps)):
            env.step(manager.ctrl)
        env.set_default_joint_values(E.build_default_joint_values())
        env.mj_forward()
        _sp = _site_pos(env, EE_SITE_R)
        print(f"[start] 复刻采集器沉降 {args.settle_steps} 步"
              f"（{time.time()-_ts0:.1f}s）→ 右末端 "
              f"{None if _sp is None else np.round(_sp, 4).tolist()}", flush=True)
        # 沉降后复检摆放：工具会落到桌面，位置会小幅变化；这里报的是**策略真正看到**的布局。
        # 判据：5 件都该离箱 0.4m 以上（训练里 5 件全在桌面）。若有一件贴近箱心，
        # _mark_pre_scoring 会把它算成"开局已达标"，开局白送分且 cycle_idx 会跳。
        if args.plant_layout != "off":
            _box_ck = _site_pos(env, TOOLBOX_SITE)
            _ts_ids = [p["site"] for p in SCORE_PHASES]
            _tp = _sites_pos(env, _ts_ids)
            _dd = [_dist(p, _box_ck) for p in _tp]
            print("[plant] 沉降后复检（距工具箱 site，m）：" + "  ".join(
                f"{TOOL_LABELS[i]}={'-' if _dd[i]!=_dd[i] else f'{_dd[i]:.3f}'}"
                for i in range(5)), flush=True)
            _close = [TOOL_LABELS[i] for i in range(5) if _dd[i] == _dd[i] and _dd[i] <= 0.25]
            print(f"[plant] {'✓ 5 件全在桌面，开局无白送分' if not _close else '!! 仍有工具贴近箱心：' + ', '.join(_close)}",
                  flush=True)


    if args.dry_run:
        # 不连相机/策略，仅测 idle 控制步墙钟，供预算分配。
        # **分开计时**：这三段里只有 env.step 是"真物理"，render 是纯开销。
        # 预算算式的分母就是这里 —— 1 env 步 = 5ms 仿真，所以
        #   能在墙钟里跑完的仿真时长 = (墙钟 / ms_per_step) * 5ms
        # 165s 墙钟 / 12.5ms = 13200 步 = 66s 仿真，而训练一集 105s
        # ⇒ **5 件工具物理上塞不进去**，天花板约 3 件。要突破只能砍 render。
        t0 = time.time(); N = 60
        t_ctrl = t_step = t_render = 0.0
        for _ in range(N):
            _a = time.time(); act = manager.run_controllers(); _b = time.time()
            env.step(act); _c = time.time()
            env.render(); _d = time.time()
            t_ctrl += _b - _a; t_step += _c - _b; t_render += _d - _c
        dt = (time.time() - t0) / N
        print(f"\n[dry-run] idle 控制步均值 {dt*1000:.1f} ms/step → 软时限 "
              f"{_deadline_text(SOFT_DEADLINE_S)}"
              + ("" if not np.isfinite(SOFT_DEADLINE_S)
                 else f" 约 {SOFT_DEADLINE_S/dt:.0f} 步"),
              flush=True)
        print(f"[dry-run] 拆解：run_controllers {t_ctrl/N*1000:5.2f} ms  "
              f"env.step {t_step/N*1000:5.2f} ms  env.render {t_render/N*1000:5.2f} ms"
              + ("" if not np.isfinite(SOFT_DEADLINE_S)
                 else f"  → 可跑的仿真时长 {SOFT_DEADLINE_S/dt*0.005:.1f}s（训练一集 105s）"),
              flush=True)
        print("[dry-run] 场景探测通过。工具初始位已记录。", flush=True)
        env.close()
        return

    # ---- 相机 + 策略 runner（评分路径才需要）----
    camera_map = E.omnipicker_camera_map(enable_wrist_l=False)
    camera_name_map: dict[str, str] = {e: k for e, (k, _port) in camera_map.items()}
    _need_cameras = not args.no_images
    _cameras: dict = {}
    _target_hw = E.DEFAULT_HW
    if _need_cameras:
        os.makedirs(E.STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(E.STREAM_TRIGGER_PATH)
        _cameras = E.bring_up_cameras(camera_map, port_timeout=30.0, frame_timeout=30.0)
        _target_hw = E.probe_camera_hw(_cameras, camera_map)
        print(f"[setup] 相机就绪 {len(_cameras)} 路 hw={_target_hw}", flush=True)
    policy_runner = E.OpenPIPolicyRunner(
        host=args.host, port=args.port, prompt="",
        camera_name_map=camera_name_map, cameras=_cameras,
        target_hw=_target_hw, use_images=not args.no_images,
    )
    print(f"[setup] 已连接策略服务 {args.host}:{args.port}", flush=True)
    # 服务端握手时就发了它训练时的 state 维度；这里硬校验，不符直接抛。
    # 维度错是**静默**的（pi0.5 的 state 是 prompt 里的一串数字文字），只能靠这道校验拦。
    E.check_server_state_dim(policy_runner.metadata, storage.state_dim)
    if _need_cameras:
        E.warmup_camera_capture(manager, env, device, _init_apply, args.camera_warmup_steps)

    if args.smoke:
        print(f"\n[smoke] camera+policy 冒烟 N={args.smoke} 步（不评分）…", flush=True)
        policy_runner.prompt = SCORE_PHASES[2]["prompt"]  # 扳手（最近 0.36m）
        t0 = time.time()
        for _ in range(args.smoke):
            obs_state = storage.build_state(storage.obs_callback(env))
            chunk = policy_runner.infer_action_chunk(obs_state)
            for ma in chunk:
                pa = E.parse_policy_action(ma)
                device.set_target(**E.action_dict_for_apply(pa))
                act = manager.run_controllers()
                env.step(act)
                env.render()
        print(f"[smoke] 完成，均 {((time.time()-t0)/args.smoke)*1000:.1f} ms/步", flush=True)
        env.close()
        return

    # ---- 起始状态对齐（--start_ref）：把右臂**位置**伺服到训练帧 0 ----
    # 为什么需要：训练是一条固定时间线，帧 0 是策略见过的唯一开局（100 集抽查极差 ≤3mm）。
    # 而 ④ 原来的起始右臂末端 (0.6822,-0.0802,0.2355) 距训练帧 0 的
    # (0.1585,-0.2792,0.4356) 有 **595mm**，右夹爪还停在 norm 0.333 而训练是 0.048(全开)
    # —— 开局整个是分布外的。分布外 → 策略在空处乱伸乱闭 → 每次"闭→开"都把 cycle_idx
    # 顶高一格 → state 告诉模型"换下一件"，于是它掠过最左边那件。见 README 二·九。
    #
    # 姿态怎么办（2026-09-17 定论，见 .runlogs/wrist_search.log）：
    #   训练帧 0 那个右臂位姿**今天的模型摆不出来**。证据：把位置锁死在已验证可达的那组
    #   关节上（位置 0.38mm），再扫腕部两关节 j5×j7 共 5184 个点，姿态误差最小只有 41.16°；
    #   让 OSC 位置+姿态一起伺服，则停在 **50.6mm / 17.9°**，且 j4=100.29°、j6=-99.85°
    #   双双顶在限位上（100.2676 / -99.6947）—— 差一点点，但确实差。
    #   所以只有两个选择：
    #     --start_align pos ：位置 0.4mm 精确，姿态差 **131.5°**   ← 默认（实测更好）
    #     --start_align pose：姿态差 17.9°，位置差 50.6mm
    #   【2026-09-17 实测定论：pos 胜，2/5 对 1/5】我原来的猜测是 pose 会更好，因为训练
    #   用了 cam_wrist_r 装在右腕上，腕部转 131° 会把腕相机甩掉，而演练里模型"每一轮都
    #   往 state 标的右边那一件去"。**这个猜测被实测否掉了**：
    #     pos  2/5，cycle_idx 1→2→3 正常推进（每轮都真的闭合→张开）
    #     pose 1/5，cycle_idx 卡在 1 整整 165 s（在 ee_tool 43mm 处闭合后抱着不放 120 s）
    #   而且 pose 模式**也**偏右（EE 最近的是 #1@11mm / #2@35mm，仍不是 #0）。
    #   结论：腕相机姿态不是主因，"偏右一把"不是姿态引起的；精确位置比姿态重要得多。
    #   偏右的真正几何原因见下面 2026-09-17 补注。
    #
    # 【2026-09-17 补注：为什么"少抓最左"—— 开局就在第三把工具的正上方】
    #   把 y 坐标排一排（y 大 = 左）：
    #     #4电工刀(右) -0.4925   #3手电筒 -0.4153   #2电工刀(左) -0.2994
    #     >>> 训练帧0 的末端 y = -0.2792 <<<   #1螺丝刀 -0.2158   #0扳手 -0.0845
    #     盒子 +0.1684
    #   训练帧 0 那个位姿，末端正好停在 **#2 和 #1 之间、更靠 #1 的位置**：
    #   离 #1 只有 63mm，离 #2 只有 20mm，而离最左边那把要抓的 **#0 扳手有 194mm**。
    #   也就是说开局手臂必须先横穿整排工具往 +y 飞 194mm 才够得到扳手，而它**在 63mm
    #   处（= 螺丝刀）就停下了**。实测每一轮都这样（EE 最近参照点）：
    #     repeat=1:  idx=1→#1@53mm(#0@112)  idx=2→#2@24mm(#1@98)  idx=3→#3@30mm(#2@118)
    #     repeat=10: idx=1→#1@26mm(#0@84)
    #   整轮整齐右移**一个工具间距(131mm)**，不是乱抓。而 state 里 goal_y 给的是对的
    #   （idx=1 时 -0.0845 = #0，idx=2 时 -0.2158 = #1，idx=3 时 -0.2994 = #2，逐轮核过），
    #   模型是**看着正确的 goal_y、跑到差一格的地方**。
    #   另：用训练集自己的 action 对照更直接 —— 训练帧 0 的命令 Δy = **-0.0631**
    #   （先往 -y 甩一下再回来），而演练 t=0.3 那帧的 state 位置只差 4.7mm，
    #   命令 Δy 却是 **+0.0176**（方向相反）。同一状态、不同动作 ⇒ 差异来自**观测**。
    #
    # ⚠ 位置：这一段**必须放在 warmup_camera_capture 之后**（见上）。
    #   2026-09-17 第一次演练就栽在这：对齐写在 warm-up 之前，warm-up 用 _init_apply
    #   逐步下发 + sleep，把手臂拉回 conf neutral、夹爪拉回 norm 0.333 —— 对齐白做。
    #   日志里 4.72mm 是真的，但回合第一帧的 state 已经是 (0.2078,-0.2936,0.4234) /
    #   夹爪 0.3333，比训练帧0 偏 55mm。开局仍是分布外 → 症状原样复发。
    _align_state: dict = {"s": None}   # 对齐刚结束时的 state，供回合开局自检比"有没有被抹掉"
    if args.start_ref and not args.no_start_servo and not args.dry_run:
        try:
            with open(args.start_ref, "r", encoding="utf-8") as f:
                _sr = json.load(f)
            _p_t = np.asarray(_sr["right_ee_pos_b"], dtype=np.float64)
            _q_t = np.asarray(_sr["right_ee_quat_b"], dtype=np.float64)
            _g = np.asarray(_sr["right_grip_norm"], dtype=np.float64)
            _p_now = _site_pos(env, EE_SITE_R)
            _pose_mode = (args.start_align == "pose")
            # ⚠ _site_pos 给的是**世界系**坐标，而 _p_t 是 base 系 —— 直接相减会得到
            # "相距 31202 mm" 这种没意义的数（2026-09-17 实测踩到）。要报距离就得
            # 拿 base 系的当前末端，也就是 state 的 7:10。
            _st_now = np.asarray(storage.build_state(storage.obs_callback(env)),
                                 dtype=np.float64).reshape(-1)
            print(f"\n[start] 起始对齐({args.start_align}) → 训练帧0 右臂位置 {np.round(_p_t, 4).tolist()}"
                  f"（当前 {np.round(_st_now[7:10], 4).tolist()}，"
                  f"相距 {_dist(_st_now[7:10], _p_t)*1000:.0f} mm）", flush=True)
            if _pose_mode:
                from scipy.spatial.transform import Rotation as _Rot
                _Rt = _Rot.from_quat(_q_t)
                print(f"    [start] 同时伺服姿态 {np.round(_q_t, 4).tolist()}"
                      f"（已知最好只能到 ~17.9°，位置会因此停在 ~50mm —— 这是取舍，不是没收敛）",
                      flush=True)
            # ---- 模式 ik：直接 teleport 关节，不用笛卡尔伺服 --------------------
            # 为什么需要它：笛卡尔伺服（pos/pose）到不了训练帧 0 那个位姿 ——
            #   pos ：位置 16.6mm / **姿态 141°**
            #   pose：位置 50.6mm / 姿态 17.9°（且 j4、j6 双双顶限位）
            # 而全关节 IK 有另一条冗余分支：**姿态 ~0° / 位置 ~0mm**。
            # 用位置代价换掉 140° 的姿态误差。腕相机(cam_wrist_r)就装在腕上，
            # 姿态差 140° 等于把腕相机的视向整个转掉 —— 而"帧 0 的 state 在 100 集里
            # 完全一致、唯一条件是图像"，所以这一项比位置重要得多。
            #
            # 2026-09-17 关键修正：**不能再读 json 里存好的 ik_joints_r**。仿真每次重建
            # 场景都会重新随机化机器人模型（每节臂 body 的四元数是随机抽的，实测同一个
            # arm_r_link1 两次等效轴角 24.62°/21.33°），存下来的关节值换一次运行就作废 ——
            # 实测写进去末端偏 593.82mm。改成**当场在 env.gym._mjModel 上现解**，
            # 同一次运行里现解现写，官方接口实测 0.00mm/0.00°。
            # json 里的值降级为兜底：现解失败（模型拿不到 / 没收敛）时才用。
            _skip_servo = False
            if args.start_align == "ik":
                _q_ik = None
                _t_ik = time.time()
                print(f"    [start] ik 模式：在**本次运行的**仿真模型上现解 IK"
                      f"（{args.ik_restarts} 次重启）…", flush=True)
                _sol = _solve_ik_sim(env, _p_t, _q_t, restarts=args.ik_restarts,
                                     seed=args.ik_seed,
                                     min_margin_deg=args.ik_min_margin_deg)
                if _sol is not None:
                    _q_ik, _dp, _da, _mg = _sol
                    print(f"    [start] 现解完成 {time.time()-_t_ik:4.1f}s："
                          f"位置 {_dp*1000:.2f}mm  姿态 {_da:.2f}°  最小限位余量 {_mg:.2f}°"
                          f"  关节(度) {np.round(np.degrees(_q_ik), 2).tolist()}", flush=True)
                elif "ik_joints_r" in _sr:
                    # 兜底：存下来的解在**本次运行**里大概率是错的（模型被重新随机化了），
                    # 但总比直接退回 pos 伺服（姿态差 141°）强，且 teleport 后还有一次实测。
                    _q_ik = np.asarray(_sr["ik_joints_r"], dtype=np.float64).reshape(7)
                    print("    [start] ⚠ 现解失败 → 退回 json 里存的 ik_joints_r。"
                          "该值**只在当初那次运行的模型上有效**，本次很可能落不到目标"
                          "（实测可偏 593mm），下面的实测残差会暴露。", flush=True)
                else:
                    print("[start] 现解失败且 start_ref 里没有 ik_joints_r → 退回 pos 伺服",
                          flush=True)
                if _q_ik is not None:
                    _jidx = [env.joint(n) for n in E.agent_conf.r_arm["joint_names"]]
                    from scipy.spatial.transform import Rotation as _Rq
                    _Rt_ik = _Rq.from_quat(_q_t)
                    _ik_bad = False   # 任一步实测读不到 → 不许再置 _skip_servo
                    for _it in range(3):
                        env.set_joint_qpos({j: np.float32(v)
                                            for j, v in zip(_jidx, _q_ik)})
                        env.mj_forward()
                        # teleport 后**立刻**实时量一次，且这一步之前不做任何 env.step：
                        # 这样才能把"teleport 本身没生效"和"生效了又被控制器拽回"分开。
                        _lv = _live_ee_b(env)
                        if _lv is None:
                            print("    [start] ⚠ 实时位姿读不到（query 失败），ik 无法闭环",
                                  flush=True)
                            _ik_bad = True
                            break
                        _lp, _lq = _lv
                        _dphi = np.degrees(np.linalg.norm(
                            (_Rt_ik.inv() * _Rq.from_quat(_lq)).as_rotvec()))
                        print(f"    [start] it{_it} teleport 后实测 位置残差 "
                              f"{_dist(_lp, _p_t)*1000:6.2f}mm  姿态残差 {_dphi:6.2f}°", flush=True)
                        # 把 OSC 目标**重新锚定到实时实测位姿**（不是缓存的那个）。
                        _a = dict(_init_apply)
                        _a["r_pos_b"] = _lp.astype(np.float32)
                        _a["r_quat_b"] = _lq.astype(np.float32)
                        _a["r_grip_ctrl"] = np.array([_g[0] * 3.0 - 1.0] * 2, dtype=np.float32)
                        _a["l_grip_ctrl"] = np.array([_g[1] * 3.0 - 1.0] * 2, dtype=np.float32)
                        device.set_target(**_a)
                        device.set_left_hold(_a["l_pos_b"], _a["l_quat_b"], _a["l_grip_ctrl"])
                        # 只跑 1 步让控制器接受新目标并 mj_forward，不再跑 30 步
                        # —— 跑得越多，OCS 的稳态误差把手臂带得越远（pos 模式实测就是这么漂的）。
                        act = manager.run_controllers()
                        env.step(act); env.render()
                    # 3 次 teleport 全部成功 → 跳过笛卡尔伺服（伺服对它没有增益，
                    # 实测 pos 16.6mm/姿态141°、pose 50.6mm/17.9°，都远差于 teleport）。
                    # ⚠ 这一行**必须**在下面 `if _skip_servo:` 之前 —— 之前一次编辑把
                    # 它误删了，导致收尾 teleport 成了死代码（跑了两轮才发现）。
                    _skip_servo = not _ik_bad
                    # 收尾再 teleport 一次。上面每一步的 `run_controllers + env.step` 都会让 OSC
                    # 把手臂从 teleport 位姿上带走一点（实测 ik5：现解 0.00mm，跑完 3 步后
                    # 官方查询 3.16mm / 2.97°）。既然 OSC 目标已经锚在 teleport 位姿上，
                    # 这里**只写 qpos、不下发、不 step**：下一次 run_controllers 的力矩就作用在
                    # 这个姿态上，而目标又是它自己，于是保持不动 —— 既精确又不会漂。
                    if _skip_servo:
                        env.set_joint_qpos({j: np.float32(v)
                                            for j, v in zip(_jidx, _q_ik)})
                        env.mj_forward()
                        _lv = _live_ee_b(env)
                        if _lv is None:
                            # ik6 上这里静默失败了（没打出下面那行）。别再不吭声：
                            # _live_ee_b 把异常吞了，这里至少报出来，否则"收尾 teleport
                            # 到底跑没跑"只能靠数日志行。
                            print("    [start] ⚠ 收尾 teleport 后实时位姿读不到"
                                  "（_live_ee_b 返回 None），回合从上面的 3 步结果起跑",
                                  flush=True)
                        if _lv is not None:
                            _lp, _lq = _lv
                            _dphi = np.degrees(np.linalg.norm(
                                (_Rt_ik.inv() * _Rq.from_quat(_lq)).as_rotvec()))
                            print(f"    [start] 收尾 teleport 后实测 位置残差 "
                                  f"{_dist(_lp, _p_t)*1000:6.2f}mm  姿态残差 {_dphi:6.2f}°"
                                  f"（这一步不 step，直接作为回合起始）", flush=True)
                            _a = dict(_init_apply)
                            _a["r_pos_b"] = _lp.astype(np.float32)
                            _a["r_quat_b"] = _lq.astype(np.float32)
                            _a["r_grip_ctrl"] = np.array([_g[0] * 3.0 - 1.0] * 2,
                                                         dtype=np.float32)
                            _a["l_grip_ctrl"] = np.array([_g[1] * 3.0 - 1.0] * 2,
                                                         dtype=np.float32)
                            device.set_target(**_a)
                            device.set_left_hold(_a["l_pos_b"], _a["l_quat_b"],
                                                 _a["l_grip_ctrl"])

            _ts = time.time()
            _last = 0.0
            # 残差追踪：伺服会**冲过最优点再往回漂**。实测（2026-09-17，d22_rev_rep1）：
            #   t=1.5s 45.8mm → t=3.0s **7.4mm** → t=4.5s 12.7 → 6.0s 14.6 → 9.1s 16.3
            # 跑满 10s 反而停在 16.6mm。命令是恒定的（每步都下发 _p_t），漂移来自
            # OSC 的稳态误差累积。所以**在最小残差处收手**比跑满划算一倍。
            _best_d = float("inf")
            _best_t = 0.0
            _worse = 0
            _q0 = None
            while not _skip_servo and (time.time() - _ts) < args.start_servo_max_s:
                _cur = np.asarray(storage.build_state(storage.obs_callback(env)),
                                  dtype=np.float64).reshape(-1)
                _a = dict(_init_apply)
                _a["r_pos_b"] = _p_t.astype(np.float32)
                # pos 模式：**冻结**进入循环那一刻的姿态作为目标。
                # 原来是每步 `= _cur[10:14]`（跟着当前姿态走）—— 那等于完全放弃姿态约束，
                # 手臂在零空间里没有回复力，于是边收敛位置边慢慢转、转着转着位置也漂走：
                # 实测 7.4mm(@3.0s) → 16.3mm(@9.1s)，命令恒定却越走越远。冻住就没了这个自由度。
                # pose 模式仍按原样较姿态到 _q_t。
                if _pose_mode:
                    _a["r_quat_b"] = _q_t.astype(np.float32)
                else:
                    if _q0 is None:
                        _q0 = np.asarray(_cur[10:14], dtype=np.float64).copy()
                    _a["r_quat_b"] = _q0.astype(np.float32)
                # 夹爪也回到训练帧0：norm 0.048 → 电机 0.048*3-1 = -0.856（张开）
                _a["r_grip_ctrl"] = np.array([_g[0] * 3.0 - 1.0] * 2, dtype=np.float32)
                _a["l_grip_ctrl"] = np.array([_g[1] * 3.0 - 1.0] * 2, dtype=np.float32)
                device.set_target(**_a)
                device.set_left_hold(_a["l_pos_b"], _a["l_quat_b"], _a["l_grip_ctrl"])
                act = manager.run_controllers()
                env.step(act)
                env.render()
                _el = time.time() - _ts
                _d = float(np.linalg.norm(_cur[7:10] - _p_t))
                _da = 0.0
                if _pose_mode:
                    try:
                        _da = float(np.degrees(
                            (_Rt * _Rot.from_quat(_cur[10:14]).inv()).magnitude()))
                    except Exception:
                        _da = float("nan")
                if _el - _last >= 1.5:
                    _last = _el
                    print(f"    [start] t={_el:4.1f}s 位置 {_d*1000:6.1f} mm"
                          + (f"  姿态 {_da:6.2f}°" if _pose_mode else ""), flush=True)
                # pose 模式下位置注定到不了 5mm（见上），只有 pos 模式才按位置判收敛
                if _d < 0.005 and (not _pose_mode or _da < 4.0):
                    break
                if _d < _best_d - 1e-4:
                    _best_d, _best_t, _worse = _d, _el, 0
                elif _d > _best_d * 1.15 + 0.0015:
                    # 持续恶化才认（40 个控制步 ≈ 0.5s），躲开单步抖动
                    _worse += 1
                    if _worse >= 40:
                        print(f"    [start] 残差已从最好 {_best_d*1000:.1f}mm(@{_best_t:.1f}s) "
                              f"恶化到 {_d*1000:.1f}mm → 提前收手", flush=True)
                        break
                else:
                    _worse = 0
            _s_fin = np.asarray(storage.build_state(storage.obs_callback(env)),
                                dtype=np.float64).reshape(-1)
            _align_state["s"] = _s_fin.copy()
            _res_p = float(np.linalg.norm(_s_fin[7:10] - _p_t))
            _res_a = float("nan")
            try:
                from scipy.spatial.transform import Rotation as _Rot2
                _res_a = float(np.degrees(
                    (_Rot2.from_quat(_q_t) * _Rot2.from_quat(_s_fin[10:14]).inv()).magnitude()))
            except Exception:
                pass
            print(f"[start] 对齐完成（{time.time()-_ts:.1f}s）：位置残差 {_res_p*1000:.2f} mm"
                  f"  姿态残差 {_res_a:.2f}°（模式 {args.start_align}，口径=缓存 obs）；"
                  f"state = {np.round(_s_fin, 4).tolist()}", flush=True)
            # 用**实时**读数交叉校验一次。缓存 obs 和实时读数不一致 ⇒ 缓存是滞后的，
            # 那"对齐完成"报的数就是假的（ik 模式第一版就报过 146mm，实际是锚错了目标）。
            _lvf = _live_ee_b(env)
            if _lvf is not None:
                _lp2, _lq2 = _lvf
                try:
                    from scipy.spatial.transform import Rotation as _Rot3
                    _a2 = float(np.degrees(np.linalg.norm(
                        (_Rot3.from_quat(_q_t).inv() * _Rot3.from_quat(_lq2)).as_rotvec())))
                except Exception:
                    _a2 = float("nan")
                _p2 = _dist(_lp2, _p_t) * 1000
                _flag = "" if abs(_p2 - _res_p * 1000) < 1.0 else "  ⚠ 与缓存口径不符"
                print(f"[start] 实时交叉校验：位置残差 {_p2:6.2f} mm  姿态残差 {_a2:6.2f}°"
                      f"（实时 query）{_flag}", flush=True)
            print(f"[start] 对照训练帧0 = {np.round(np.asarray(_sr['state22'], dtype=np.float64), 4).tolist()}",
                  flush=True)
        except Exception as _se:
            print(f"[start] 起始对齐失败（继续跑）: {_se}", flush=True)

    # ---- 评分生命周期 ----
    scorer = None
    if not args.no_score:
        from orca_scorer_client import ScorerClient
        # timeout 必须显式给大值。ScorerClient 的默认是 **5.0 秒**，而
        # /api/attempt/finish 要逐帧回溯整段 attempt（上千帧）才能算分，
        # 5 秒必然超时 → finish() 返回的是 {"error": "…timed out"}，
        # 脚本于是打出 `score: None / None`。**分其实算出来了、也落盘了**
        # （在 site-packages/.orca_comp_scorer/summary_att_<id>.json，隐藏目录），
        # 只是这一趟拿不到 —— 2026-09-11 那次正式评分就栽在这，白跑一次。
        scorer = ScorerClient(
            team_id=args.team_id, team_token=args.team_token, robot_id=args.robot_id,
            timeout=180.0,
        )
    if scorer:
        info = scorer.start_attempt("task3_tool_storage", prompt="整理工具(5件逐一收纳)")
        print(f"[scorer] attempt started: {info.get('attempt_id')}", flush=True)
    else:
        print("[scorer] 演练模式（无评分上报）", flush=True)

    _dump_frames(_cameras, "attempt_start", args.frames_dir)
    t_start = time.time()

    # 原始推理输出转储（--dump_chunks）。只写文件、不改控制流；异常时也不让它拖垮主流程。
    _dump_fh = None
    if args.dump_chunks:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_chunks)), exist_ok=True)
        _dump_fh = open(args.dump_chunks, "w", encoding="utf-8")
        print(f"[dump] 原始推理输出 -> {args.dump_chunks}", flush=True)

    placed: dict[str, bool] = {}
    phase_logs: list[dict] = []
    attempt_started = bool(scorer)
    finished = False
    # 回合进行中的"半截记录"。手动 Ctrl-C（现在是不限时模式下的**唯一**人工终止手段）
    # 会从回合函数里抛出去，日志对象根本来不及构造 —— 而演练的全部价值就在那份 jsonl。
    # 所以循环里每轮把当前进度写进这里，收尾时若 phase_logs 是空的就补一份出来。
    partial: dict = {}

    # ------------------------------------------------------------------
    # 单回合模式（默认）
    # ------------------------------------------------------------------
    def _run_single_episode() -> tuple[dict, list]:
        """一整集连续收纳 5 件工具 —— 与训练数据结构一致。

        为什么不再切 5 个阶段（2026-09-17）：
          · 训练是**一集 2101 帧**：右夹爪"闭→开"5 次，state 第22维 cycle_idx 从 1 涨到 6，
            第 5 次释放后 demo 还静止 ~6.75s。pi0.5 的 18 维动作里**没有任何 done 位** ——
            模型能停下来的唯一依据就是 idx，它必须自己涨到 6，模型才会做出"收完了"。
          · 旧写法每段换一个**模型从没训练过的裸工具名**当 prompt（训练只有 3 句整场景话、
            全不含工具名）、每段一个独立时间预算，而 reset_cycle() 只在 5 段之前调一次
            → 阶段状态机和周期计数器不同步，idx 到不了 6。
          · 实测（.runlogs/d22_no_score_chunks.jsonl）三次 idx 推进（1→2@3.9s、2→3@72.0s、
            3→4@129.8s）全都发生在**离最近工具 161/252/234 mm 的半空中**，没有一次是真抓取；
            而命令的末端目标**从头到尾没靠近过最左那件扳手**（最近 150mm，之后再没更近）。
            也就是说：不是模型"选择"跳过，是 state 里的目标被顶走了。
          · 评分器判的是**场景最终状态与整段 attempt 的时间线**，与我们的循环结构无关。
        """
        policy_runner.prompt = args.episode_prompt
        print("\n[episode] === 单回合连续收纳 ===", flush=True)
        print(f"[episode] prompt = {args.episode_prompt!r}（训练语料原句）", flush=True)
        print(f"[episode] 停止条件: cycle_idx>={args.stop_cycle_idx}（模型学到的『干完了』，再给 "
              f"{args.stop_settle_s:.0f}s 让最后一件落稳）| 5 件全在箱内且夹爪张开并保持 "
              f"{CONFIRM_S}s | 挂钟软时限 {_deadline_text(SOFT_DEADLINE_S)}"
              f"（不限时 = 手动 Ctrl-C 停）", flush=True)
        _dump_frames(_cameras, "episode_start", args.frames_dir)

        tool_sites = [p["site"] for p in SCORE_PHASES]
        tname = {p["site"]: p["prompt"] for p in SCORE_PHASES}
        site0 = _sites_pos(env, tool_sites)
        box0 = _site_pos(env, TOOLBOX_SITE)
        # 箱子内部范围标定一次（判"真的放进去几件"）。--progress_gate grip 时不需要。
        global _IN_BOX_MAX, _HOLD_SINCE, _PRE_SKIP
        _IN_BOX_MAX = 0                 # 每试清零，别把上一试的进度带过来
        _HOLD_SINCE = {}
        if args.progress_gate == "box":
            _calibrate_box_inner(env)
        # grip 口径下 _PRE_SKIP 根本不被读（idx 只由开爪边沿驱动），跳过这段——
        # 否则它按 OFFICIAL_SCORE_DIST 判一次并打印"开局已达标"，在 grip 下纯属误导。
        if args.progress_gate == "grip":
            _PRE_SKIP = set()
        elif args.progress_baseline == "initial":
            _PRE_SKIP = _mark_pre_scoring(env, tool_sites, box0, args.progress_gate)
        else:
            _PRE_SKIP = set()
            print("[progress] --progress_baseline none：开局基准不扣除"
                  "（本场景手电筒开局就在箱边，cycle_idx 会开局 +1）", flush=True)
        if args.progress_gate in ("box", "official"):
            storage.set_progress_override(0)

        def _state() -> np.ndarray:
            """取 state：**先**刷新进度覆盖，再 build_state（顺序不能反）。"""
            if args.progress_gate == "official":
                _n = _count_scoring(env, tool_sites, box0, step)
                if _n is not None:
                    storage.set_progress_override(_n)
            elif args.progress_gate == "box":
                _n = _count_in_box(env, tool_sites, box0)
                if _n is not None:
                    storage.set_progress_override(_n)
            return np.asarray(storage.build_state(storage.obs_callback(env)),
                              dtype=np.float64).reshape(-1)

        step = 0
        last_log = time.time()
        last_cyc = None
        cyc_trace: list[dict] = []
        cyc_stop_t: float | None = None
        all_near_since: float | None = None
        done_reason: str | None = None
        min_ee_tool = float("inf")
        min_ee_base = float("inf")

        # 开局自检 —— 两个参照，作用不同：
        #   vs `_align_state`（对齐刚结束时）→ 检"对齐有没有被后面某步抹掉"。
        #     2026-09-17 就是栽在这：对齐在 warm-up 之前做，warm-up 把它拉回 neutral，
        #     这里本该一眼看出来的 55mm 漂移，当时没有这道自检就白跑一轮。
        #   vs 训练帧 0 → 检"开局离分布有多远"。pos 模式下只剩姿态偏（摆不到）；
        #     pose 模式下位置会偏 ~50mm，那是已知取舍，不是故障。
        try:
            _st0 = _state()
            if _align_state.get("s") is not None:
                _dr = np.abs(_st0 - np.asarray(_align_state["s"], dtype=np.float64))
                print(f"[episode] 开局漂移（vs 对齐刚结束）：位置 "
                      f"{np.linalg.norm(_st0[7:10]-np.asarray(_align_state['s'])[7:10])*1000:.1f} mm，"
                      f"夹爪 {_st0[16]:.4f}，最大维度偏差 {_dr.max():.4f}"
                      f"{'  ← 对齐被抹掉了！' if _dr.max() > 0.02 else ''}", flush=True)
            with open(args.start_ref, "r", encoding="utf-8") as _f:
                _s0 = np.asarray(json.load(_f)["state22"], dtype=np.float64)
            _dd = np.abs(_st0 - _s0)
            _bad = " ".join(f"[{i}]{v:.4f}" for i, v in enumerate(_dd) if v > 0.02) or "无"
            _pos = float(np.linalg.norm(_st0[7:10] - _s0[7:10]))
            print(f"[episode] 开局自检：位置偏 {_pos*1000:.1f} mm，夹爪 {_st0[16]:.4f}"
                  f"（训练 {_s0[16]:.4f}），cycle_idx {_st0[21]:.0f}"
                  f"；|Δ|>0.02 的维度: {_bad}", flush=True)
        except Exception as _e0:
            print(f"[episode] 开局自检跳过: {_e0}", flush=True)

        while (time.time() - t_start) < SOFT_DEADLINE_S:
            obs_state = _state()
            st = np.asarray(obs_state, dtype=np.float64).reshape(-1)
            cyc_idx = int(round(st[21])) if st.size >= 22 else -1
            cyc_dist = float(st[20]) if st.size >= 22 else float("nan")
            goal_y = float(st[19]) if st.size >= 22 else float("nan")
            if cyc_idx != last_cyc:
                cyc_trace.append({"t": round(time.time() - t_start, 2), "cycle_idx": cyc_idx,
                                  "cycle_dist": round(cyc_dist, 4), "goal_y": round(goal_y, 4)})
                ee_n = _site_pos(env, EE_SITE_R)
                _c5 = _sites_pos(env, tool_sites)
                dtxt = "  ".join(f"#{j}{tname[tool_sites[j]]}:{_dist(ee_n, _c5[j])*1000:.0f}mm"
                                 for j in range(len(tool_sites)))
                print(f"    [cycle] t={time.time()-t_start:6.1f}s  idx {last_cyc} -> {cyc_idx}"
                      f"   cycle_dist={cyc_dist:.3f}  goal_y={goal_y:+.3f}   末端离各工具: {dtxt}",
                      flush=True)
                last_cyc = cyc_idx

            chunk = policy_runner.infer_action_chunk(obs_state)

            if _dump_fh is not None:
                cur5 = _sites_pos(env, tool_sites)
                _ee_w = _site_pos(env, EE_SITE_R)
                _bs_w = _body_pos(env, ROBOT_BASE_BODY)
                _bx_w = _site_pos(env, TOOLBOX_SITE)
                _near = None
                if _ee_w is not None:
                    _ds = [(_dist(_ee_w, p), p) for p in cur5 if p is not None]
                    _near = min(_ds, key=lambda x: x[0])[1] if _ds else None
                try:
                    _dump_fh.write(json.dumps({
                        "t": round(time.time() - t_start, 3),
                        "phase_t": round(time.time() - t_start, 3),
                        "prompt": args.episode_prompt, "step_id": "episode", "step": step,
                        "cycle_idx": cyc_idx, "cycle_dist": cyc_dist, "goal_y": goal_y,
                        "state": [float(x) for x in st],
                        "chunk": [[float(x) for x in np.asarray(a).reshape(-1)[:18]] for a in chunk],
                        "ee_world": None if _ee_w is None else [float(x) for x in _ee_w],
                        "base_world": None if _bs_w is None else [float(x) for x in _bs_w],
                        "tool_world": None if _near is None else [float(x) for x in _near],
                        "tools_world": [None if p is None else [float(x) for x in p] for p in cur5],
                        "box_world": None if _bx_w is None else [float(x) for x in _bx_w],
                    }, ensure_ascii=False) + "\n")
                    _dump_fh.flush()
                except Exception as _e:
                    print(f"[dump] 写失败: {_e}", flush=True)

            for ma in chunk:
                if (time.time() - t_start) >= SOFT_DEADLINE_S:
                    break
                pa = E.parse_policy_action(ma)
                device.set_target(**E.action_dict_for_apply(pa))
                for _ in range(args.action_repeat):
                    act = manager.run_controllers()
                    env.step(act); env.render(); step += 1
                if time.time() - last_log >= 3.0:
                    last_log = time.time()
                    ee = _site_pos(env, EE_SITE_R)
                    base = _body_pos(env, ROBOT_BASE_BODY)
                    cur5 = _sites_pos(env, tool_sites)
                    ds = [(_dist(ee, p), p) for p in cur5 if p is not None]
                    eet = min((d for d, _ in ds), default=float("nan"))
                    if eet == eet:
                        min_ee_tool = min(min_ee_tool, eet)
                    eeb = _dist(ee, base)
                    if eeb == eeb:
                        min_ee_base = min(min_ee_base, eeb)
                    n_in = sum(1 for p in cur5 if _dist(p, box0) <= PLACED_DIST)
                    _cb = getattr(policy_runner, "cam_builder", None)
                    cam_s = ""
                    if _cb is not None:
                        _bri = ",".join(f"{k.split('_')[1][:4]}={v:5.1f}"
                                        for k, v in _cb.last_brightness.items())
                        cam_s = f" cam[{_bri}] miss={sum(_cb.miss_counts.values())}"
                    print(f"    [t={time.time()-t_start:5.1f}s ep] idx={cyc_idx} 入箱={n_in}/5 "
                          f"ee_tool={eet:.3f} grip_m={float(np.asarray(device.r_grip_ctrl)[0]):+.2f} "
                          f"step={step}{cam_s}", flush=True)

            # ---- 每个 chunk 判一次停 ----
            cur5 = _sites_pos(env, tool_sites)
            n_in = sum(1 for p in cur5 if _dist(p, box0) <= PLACED_DIST)
            # 半截记录：Ctrl-C 时 finally 靠它补写 jsonl（不限时模式下人工停止是常态）
            partial.update({
                "prompt": args.episode_prompt, "step_id": "episode",
                "done": False, "reason": "手动中断（Ctrl-C）",
                "cycle_idx_end": last_cyc, "cycle_trace": list(cyc_trace),
                "elapsed_total": round(time.time() - t_start, 1), "steps": step,
                "soft_deadline_s": (None if not np.isfinite(SOFT_DEADLINE_S)
                                    else round(float(SOFT_DEADLINE_S), 1)),
                "time_limited": bool(np.isfinite(SOFT_DEADLINE_S)),
                "action_repeat": int(args.action_repeat),
                "plant_layout": args.plant_layout,
                "n_in_box": int(n_in),
                "placed": {tname[p["site"]]: bool(_dist(cur5[j], box0) <= PLACED_DIST)
                           for j, p in enumerate(SCORE_PHASES)},
                "min_ee_tool": (None if not np.isfinite(min_ee_tool) else round(min_ee_tool, 4)),
                "min_ee_base": (None if not np.isfinite(min_ee_base) else round(min_ee_base, 4)),
                "partial": True,
            })
            if n_in >= len(tool_sites) and _grip_open(device):
                all_near_since = time.time() if all_near_since is None else all_near_since
                if (time.time() - all_near_since) >= CONFIRM_S:
                    done_reason = f"物理停止：5/5 件已在箱内且夹爪张开（保持 {CONFIRM_S}s）"
                    break
            else:
                all_near_since = None
            # "模型说干完了"要有一件真的入箱才认。
            # 2026-09-20：--progress_gate 默认改回 grip 后，idx 完全由模型自己开爪驱动 ——
            # 空开 5 次爪就能把 cycle_idx 顶到 6。若此时照旧收摊，盘上留下的记录是
            # "模型学到的停止"，而箱里一件都没有，等于什么都没诊断到（不限时演练模式
            # 尤其亏：本来就该继续看着它怎么失败）。官方给分口径不受影响 ——
            # 那条路走 SOFT_DEADLINE_S，不靠这个分支。
            if cyc_idx >= args.stop_cycle_idx and n_in >= 1:
                if cyc_stop_t is None:
                    cyc_stop_t = time.time()
                    print(f"    [stop] cycle_idx 已达 {cyc_idx}（模型学到的『干完了』）"
                          f"且箱内已有 {n_in}/5，再跑 {args.stop_settle_s:.0f}s 让最后一件落稳…",
                          flush=True)
                elif (time.time() - cyc_stop_t) >= args.stop_settle_s:
                    done_reason = (f"模型学到的停止：cycle_idx={cyc_idx} >= {args.stop_cycle_idx}"
                                   f"（箱内 {n_in}/5，再等 {args.stop_settle_s:.0f}s）")
                    break
            elif cyc_idx >= args.stop_cycle_idx:
                if cyc_stop_t is None:
                    cyc_stop_t = time.time()
                    print(f"    [stop] ⚠ cycle_idx 已到 {cyc_idx} 但箱内仍 0/5 —— "
                          f"**不收摊**（idx 是模型自己开爪推的，空开爪也算）。"
                          f"继续跑，看它到底在干什么。", flush=True)

        placed_out = {p["site"]: (_dist(_site_pos(env, p["site"]), box0) <= PLACED_DIST)
                      for p in SCORE_PHASES}
        el = time.time() - t_start
        _dump_frames(_cameras, "episode_end", args.frames_dir)
        print(f"\n[episode] 结束：{done_reason or _loop_exit_reason(SOFT_DEADLINE_S)}"
              f"（总用时 {el:.0f}s，{step} 步，cycle_idx 终值 {last_cyc}）", flush=True)
        # 官方口径估分：tasks.yaml 里每件工具 7 分，score_curve 是
        #   0m→1.0 / 0.25m→0.8571 / >0.2501m→0，且还要在箱心 0.3m 内连续待够 1.0s。
        # 这里用**最后一次观测到的距离**算 curve（严格说引擎是在判成功那一刻取值，
        # 演练里没有那套逐帧判定，所以这是上界估计，只作对照，不是官方分）。
        official = 0.0
        for j, p in enumerate(SCORE_PHASES):
            p_now = _site_pos(env, p["site"])
            d = _dist(p_now, box0)
            if d <= 0.0:
                ratio = 1.0
            elif d <= OFFICIAL_SCORE_DIST:
                ratio = 1.0 + (0.8571 - 1.0) * (d / OFFICIAL_SCORE_DIST)
            else:
                ratio = 0.0
            official += 7.0 * ratio
            print(f"    {p['prompt']:<10} 入箱={'✓' if placed_out[p['site']] else '✗'}  "
                  f"dist_box={d:.3f}  "
                  f"位移={_dist(p_now, site0[j]):.3f}m  "
                  f"官方curve={7.0*ratio:.2f}/7  "
                  f"cycle_idx 变化点={[c['cycle_idx'] for c in cyc_trace]}", flush=True)
        print(f"    [官方口径估分] 5 件工具合计 {official:.2f}/35（不含 reach_tool_table 的 5 分；"
              f"上界估计，未算 0.3m/1.0s 持续门槛）", flush=True)
        logs = [{"prompt": args.episode_prompt, "step_id": "episode",
                 "done": bool(done_reason), "reason": done_reason,
                 "cycle_idx_end": last_cyc, "cycle_trace": cyc_trace,
                 "elapsed_total": round(el, 1), "steps": step,
                 # 把生效的时限**写进记录**：否则事后只看 elapsed_total 分不清
                 # "跑到点被掐" 和 "不限时下自己停了" —— 2026-09-20 就被这个坑到过。
                 "soft_deadline_s": (None if not np.isfinite(SOFT_DEADLINE_S)
                                     else round(float(SOFT_DEADLINE_S), 1)),
                 "time_limited": bool(np.isfinite(SOFT_DEADLINE_S)),
                 "action_repeat": int(args.action_repeat),
                 "plant_layout": args.plant_layout,
                 "placed": {tname[k]: v for k, v in placed_out.items()},
                 "min_ee_tool": round(min_ee_tool, 4), "min_ee_base": round(min_ee_base, 4)}]
        partial.clear()          # 正常返回：不留"半截记录"，免得 finally 再写一份
        return placed_out, logs

    try:
        if args.phased:
            remaining = [p for p in SCORE_PHASES]
        else:
            _placed, _logs = _run_single_episode()
            placed.update(_placed)
            phase_logs.extend(_logs)
            remaining = []
        if args.only_phases:
            remaining = [p for p in remaining if p["step_id"] in args.only_phases]
            print(f"[run] 过滤后只跑: {[p['step_id'] for p in remaining]}", flush=True)
        while remaining:
            ph = remaining[0]
            remaining = remaining[1:]
            t_ph0 = time.time()
            if args.per_phase_max_s and args.per_phase_max_s < 999:
                budget = float(args.per_phase_max_s)
            elif not np.isfinite(SOFT_DEADLINE_S):
                # 不限时（演练）：单阶段预算原本是按"剩余时限均分"算出来的，现在没参照了。
                # **不能让它变成 inf** —— 那样第一阶段永远不会让位给第二阶段，
                # 5 阶段模式会静默退化成"只做第一件"（而且不报错，最难查）。
                # 退到 --per_phase_max_s 的默认值：有限、够长、但阶段还能推进。
                budget = float(args.per_phase_max_s)
                print(f"[phase] 软时限不限时，单阶段预算退到 --per_phase_max_s="
                      f"{budget:.0f}s（只为让阶段能推进，不是终止条件）", flush=True)
            else:
                budget = max(8.0, (SOFT_DEADLINE_S - (time.time() - t_start)) / max(len(remaining) + 1, 1) - 3.0)
            print(f"\n[phase] === {ph['prompt']} ({ph['step_id']}) 预算 {budget:.0f}s ===", flush=True)
            _dump_frames(_cameras, f"phase_{ph['prompt']}", args.frames_dir)

            policy_runner.prompt = ph["prompt"]
            step = 0
            phase_done = False
            near_since: float | None = None
            last_log = time.time()
            holding = False
            min_ee_tool = float("inf")  # 诊断：右手 EE 与当前工具的最短距离
            min_ee_base = float("inf")  # 诊断：右手 EE 与机器人底盘的最短距离

            while (time.time() - t_ph0) < budget and (time.time() - t_start) < SOFT_DEADLINE_S:
                obs_state = _state()
                chunk = policy_runner.infer_action_chunk(obs_state)
                if _dump_fh is not None:
                    # 转储在这里而不是在内层 (每条动作) —— 决策是"每次推理一次"的，
                    # 逐条动作转储只会把同一个 chunk 重复 20 遍，白白撑大文件。
                    _ee_w = _site_pos(env, EE_SITE_R)
                    _bs_w = _body_pos(env, ROBOT_BASE_BODY)
                    _tl_w = _site_pos(env, ph["site"])
                    _bx_w = _site_pos(env, TOOLBOX_SITE)
                    try:
                        _dump_fh.write(json.dumps({
                            "t": round(time.time() - t_start, 3),
                            "phase_t": round(time.time() - t_ph0, 3),
                            "prompt": ph["prompt"],
                            "step_id": ph["step_id"],
                            "step": step,
                            "state": [float(x) for x in np.asarray(obs_state).reshape(-1)],
                            "chunk": [[float(x) for x in np.asarray(a).reshape(-1)[:18]] for a in chunk],
                            "ee_world": None if _ee_w is None else [float(x) for x in _ee_w],
                            "base_world": None if _bs_w is None else [float(x) for x in _bs_w],
                            "tool_world": None if _tl_w is None else [float(x) for x in _tl_w],
                            "box_world": None if _bx_w is None else [float(x) for x in _bx_w],
                        }, ensure_ascii=False) + "\n")
                        _dump_fh.flush()
                    except Exception as _e:  # 转储只是诊断，绝不能让它打断评测
                        print(f"[dump] 写失败: {_e}", flush=True)
                released_in_chunk = False
                for ma in chunk:
                    if (time.time() - t_ph0) >= budget or (time.time() - t_start) >= SOFT_DEADLINE_S:
                        break
                    pa = E.parse_policy_action(ma)
                    device.set_target(**E.action_dict_for_apply(pa))
                    for _ in range(args.action_repeat):
                        act = manager.run_controllers()
                        _, _, _, truncated, _ = env.step(act)
                        env.render()
                        step += 1
                    holding = not _grip_open(device)
                    released_in_chunk = _grip_open(device)

                    cur = _site_pos(env, ph["site"])
                    b = _site_pos(env, TOOLBOX_SITE)
                    d = _dist(cur, b)
                    ee = _site_pos(env, EE_SITE_R)
                    base = _body_pos(env, ROBOT_BASE_BODY)
                    ee_tool = _dist(ee, cur)
                    ee_base = _dist(ee, base)
                    if ee_tool < min_ee_tool:
                        min_ee_tool = ee_tool
                    if ee_base < min_ee_base:
                        min_ee_base = ee_base
                    if time.time() - last_log >= 3.0:
                        last_log = time.time()
                        ee_s = "n/a" if ee is None else f"({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f})"
                        b_s = "n/a" if base is None else f"({base[0]:.2f},{base[1]:.2f})"
                        # 相机存活诊断：断流时上游会静默喂全黑图，只看末端位姿
                        # 分不清"策略不动"还是"相机死了"，所以把亮度和丢帧数带上。
                        _cb = getattr(policy_runner, "cam_builder", None)
                        cam_s = ""
                        if _cb is not None:
                            _bri = ",".join(
                                f"{k.split('_')[1][:4]}={v:5.1f}" for k, v in _cb.last_brightness.items()
                            )
                            _miss = sum(_cb.miss_counts.values())
                            cam_s = f" cam[{_bri}] miss={_miss}"
                        print(f"    [t={time.time()-t_ph0:5.1f}s {ph['prompt']}] "
                              f"dist_box={d:.3f} ee_tool={ee_tool:.3f} "
                              f"ee={ee_s} base_xy={b_s} "
                              f"grip_m={float(np.asarray(device.r_grip_ctrl)[0]):+.2f} step={step}{cam_s}",
                              flush=True)
                    if d <= PLACED_DIST and released_in_chunk:
                        near_since = time.time() if near_since is None else near_since
                        if (time.time() - near_since) >= CONFIRM_S:
                            phase_done = True
                            break
                    else:
                        near_since = None
                if phase_done:
                    break

            # 预算耗尽但夹爪仍闭合(可能抓着工具)：宽限到释放，避免把工具带去下一件
            if not phase_done and holding and (time.time() - t_start) < SOFT_DEADLINE_S:
                print(f"    [{ph['prompt']}] 夹爪仍合，宽限至张开…", flush=True)
                g0 = time.time()
                while (time.time() - g0) < 12.0 and (time.time() - t_start) < SOFT_DEADLINE_S and not _grip_open(device):
                    obs_state = _state()
                    chunk = policy_runner.infer_action_chunk(obs_state)
                    for ma in chunk:
                        pa = E.parse_policy_action(ma)
                        device.set_target(**E.action_dict_for_apply(pa))
                        for _ in range(args.action_repeat):
                            act = manager.run_controllers()
                            env.step(act); env.render(); step += 1
                        if _grip_open(device):
                            break
                # 张开后再给一点点时间让工具落稳
                if _grip_open(device):
                    for _ in range(int(2.0 / 0.02)):
                        act = manager.run_controllers()
                        env.step(act); env.render()

            placed[ph["site"]] = phase_done
            el = time.time() - t_start
            phase_logs.append({"prompt": ph["prompt"], "step_id": ph["step_id"], "done": phase_done,
                               "elapsed_total": round(el, 1), "steps": step})
            print(f"[phase] {ph['prompt']}: {'已放入✓' if phase_done else '未完成'} "
                  f"(总用时 {el:.0f}s, {step} 步) 诊断[min_ee_tool={min_ee_tool:.3f}m "
                  f"min_ee_base={min_ee_base:.3f}m]",
                  flush=True)

            # 顺带统计当前各工具是否已在箱内
            for p2 in SCORE_PHASES:
                cur2 = _site_pos(env, p2["site"])
                if cur2 is not None and _dist(cur2, box) <= PLACED_DIST:
                    placed[p2["site"]] = True

            if (time.time() - t_start) >= SOFT_DEADLINE_S:
                print("[phase] 软时限到，停止。", flush=True)
                break

        if scorer and attempt_started and not finished:
            summary = scorer.finish()
            finished = True
            print(f"\n[scorer] score: {summary.get('score')} / {summary.get('max_score')}", flush=True)
            for r in summary.get("step_results", []):
                print(f"  {r.get('step_id')}: {'PASS' if r.get('success') else 'FAIL'}", flush=True)
    finally:
        if scorer and attempt_started and not finished:
            try:
                summary = scorer.finish()
                finished = True
                print(f"\n[scorer] (收尾) score: {summary.get('score')} / {summary.get('max_score')}", flush=True)
            except Exception as fe:
                print(f"[scorer] 收尾 finish 失败: {fe}", flush=True)
        # 让评分客户端走**优雅关停**：它内部会 POST /api/shutdown，等服务把
        # 录屏上传完（最多 90s）再杀本地评分服务子进程。
        # 这一步必须在下面的 os._exit(0) 之前显式做，原因是：
        #   ScorerClient 是靠 `atexit.register(self._stop_local_service)` 来关停的
        #   （orca_scorer_client/__init__.py:420），而 **os._exit() 不跑 atexit**。
        # 漏掉它的实测后果（2026-09-11）：本地有分、中央服务器一条上传记录都没有，
        # 5.2MB 录屏从没上传成功 —— 也就是只有本地分、没有官方分。
        # 这里原来只有一句 `time.sleep(25)` 硬等，那治不了本（进程一 os._exit 就断）。
        # 客户端自己起的服务才会被它停；复用已有服务时它是空操作（有 _started_by_us 守卫），安全。
        if scorer is not None:
            try:
                scorer._stop_local_service()   # 无公开 API，只能用这个私有方法
            except Exception as se:
                print(f"[scorer] 优雅关停失败: {se}", flush=True)
        if args.jsonl and phase_logs:
            Path(args.jsonl).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in phase_logs),
                                        encoding="utf-8")
        elif args.jsonl and partial:
            # 回合没走到 return（手动 Ctrl-C / 异常）——补写半截记录。
            # 不加这一段的话，不限时模式下人工停止 = 那次演练什么记录都不留，
            # 而那正是我们最想看的那一次（模型跑了多久、到几件、卡在哪）。
            partial.setdefault("partial", True)
            partial.setdefault("done", False)
            partial.setdefault("reason", "手动中断（Ctrl-C）")
            partial.setdefault("soft_deadline_s",
                               None if not np.isfinite(SOFT_DEADLINE_S)
                               else round(float(SOFT_DEADLINE_S), 1))
            partial.setdefault("time_limited", bool(np.isfinite(SOFT_DEADLINE_S)))
            Path(args.jsonl).write_text(json.dumps(partial, ensure_ascii=False),
                                        encoding="utf-8")
            print(f"\n[episode] 已补写半截记录 -> {args.jsonl}"
                  f"（{partial.get('steps')} 步，{partial.get('elapsed_total')}s，"
                  f"入箱 {partial.get('n_in_box')}/5，cycle_idx={partial.get('cycle_idx_end')}）",
                  flush=True)
        if _dump_fh is not None:
            try: _dump_fh.close()
            except Exception: pass
        if _cameras:
            try: E.close_cameras(_cameras)
            except Exception: pass
        try: env.stop_save_video()
        except Exception: pass
        try: env.close()
        except Exception: pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[task3] 中断", flush=True)
    except Exception:
        import traceback; traceback.print_exc()
    finally:
        os._exit(0)
