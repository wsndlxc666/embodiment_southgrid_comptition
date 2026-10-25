"""G1 OmniPicker · Clear-All 随机化采集 → LeRobot v2.1。

真实部署任务(用户确认 2026-09-06)：桌面条带上摆工具，指令是**一整句全收、不点名**，
要求机器人**先视觉识别、然后逐把**抓起放入旁边的工具箱。

2026-09-23 改：原来 181 集是**每集必种 5 把、必抓 5 把、顺序永远是 0→1→2→3→4**，
唯一变量是几毫米的布局抖动 —— 181 集本质是同一条轨迹抖了 181 次，模型学到的是
"第几把"这个位置先验，不是"看见什么抓什么"。现在改成**四个随机**：

  ① **抓取顺序随机**：每集对要抓的那几把重新排列(混入从左到右/从右到左/随机)，
     模型无法再用"第几把"预测，只能看场景。
  ② **每集抓几把随机**：k ∈ [--k_min, --k_max]，有的集只抓 1 把、有的抓满 5 把。
     评测是 5 把，所以 k 的分布向大 k 倾斜(见 --k_weights)。
  ③ **场上放几把 与 抓几把 解耦**(--plant_mode)：
       all   — 永远种 5 把，只抓其中 k 把(其余原地不动)；
       rand  — 种 m 把(m ∈ [k,5])，其余几把搬到身后 1.2m 的地上藏起来；
       mixed — (默认)每集按 --plant_all_frac 掷一次走哪支。实测最终"场上 5 把"
               的集占 ≈0.80、1~4 把占 ≈0.20 —— 前者与评测场景同分布，
               后者逼模型看"场上到底有什么"而不是无条件输出 5 次抓取。
  ④ **起始位姿随机**：--init_jitter_deg 给双臂初始关节加小扰动，不再是同一个开局。

另外 `--tool_weights` 让**螺丝刀(index 1)被抽中的概率更高**，从而帧数更多
(用户要求：尽可能让螺丝刀多一些数据量)。

**逐帧 tool_id**：每帧写一列 `tool_id`(0..4，撤离/等待段 = -1)，由构造决定、无需感知。
它是**标签不是输入** —— 评测时"这是哪把工具"正是模型要从视觉里推出来的东西，
当输入喂进去等于把答案递给模型。它只进损失(分工具加权)。

每集：
  1. 采样本集计划(种哪几把/抓哪几把/什么顺序/初始位姿)；
  2. 种回标称槽位 ± 抖动；未种的搬离桌面；
  3. 指令 = 恒定全收词族(3 个同义句轮换，不含任何工具名)；
  4. 按采样的顺序逐把 pick+place 串成一集，每把结束登记 place_arm_at →
     设备逐把做"EE 出整箱矩形→接触白名单(箱内面/已入箱工具)"验证；
  5. 放箱防堆叠：逐把放箱 xy 在箱底内侧标定矩形内错开(--no_box_spread 可关)；
  6. 任一工具放置未通过 → 整集 clear_data 重采；连续 max_attempts 未成功则放弃换布局重开。

抓取/放箱路点：抓取侧(wp0/1)平移 Delta 到该把实际落点(含 yaw 抖动的静置落位)，
放箱侧路点保持模板，只加逐序箱内错位偏移。复用 tool 采集器的设备/几何/控制器。

**退化回旧行为**：`--k_min 5 --k_max 5 --plant_mode all --fixed_order --init_jitter_deg 0`
就是 2026-09-23 之前的固定 5 把左→右采集。

帧数：单把 ≈ 400 帧，一集 ≈ k×400 + 撤离/等待。输出：--lerobot_out。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

OPENPI_DATA_ROOT = os.path.normpath(
    os.path.join(base_dir, "..", "..", "..", "..", "..", "openpi", "data")
)

# 复用工具整理采集脚本里的设备/路点/几何/判定函数与常量。
import g1_omnipicker_collection_scripted_tool_lerobot as _T

import data_collection_scripted as scripted
from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_clear5_lerobot_stream"

orca_logger = get_orca_logger(
    name="G1Clear5Scripted",
    log_file="g1_omnipicker_collect_clear5_lerobot.log",
    max_bytes=20 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)

_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

TOOL_NAMES = _T._TOOL_NAMES                 # ["扳手","螺丝刀","电工刀(左)","手电筒","电工刀(右)"]
TOOL_JOINTS = _T._TOOL_BODY_JOINT_NAMES
TOOL_REF_POS_B = _T._TOOL_REFERENCE_POS_B   # base 系标称槽位(≈部署默认摆放)
TOOL_REF_QUAT_XYZW_B = _T._TOOL_REFERENCE_QUAT_XYZW_B

# 顺序索引 0..4 = base 系 -Y 递增 = 从左到右(扳手→螺丝刀→电工刀(左)→手电筒→电工刀(右))。
# 与评分脚本 SCORE_PHASES(按名分阶段的官方步序)无关，不要混淆。
CLEAR_ORDER = [0, 1, 2, 3, 4]   # 仅 --fixed_order 时用；随机化后每集现采

SCREWDRIVER = 1                 # TOOL_NAMES[1] = "螺丝刀"

# 恒定全收指令词族(不含工具名)。训练 task 逐集在这几条里轮换；
# 评测时取同一条，保证 prompt 完全一致。
CLEAR_PROMPTS = [
    "把桌面上的工具都收进工具箱",
    "把所有工具从左到右放回工具箱",
    "把工具全部收进箱子里",
]

# 放置验证超时 / 末段撤离(帧数)，同 single_place。
_PLACE_CHECK_TIMEOUT_S = 2.5
_PLACE_RETREAT_STEPS = 50


def _clear_prompt(rng: np.random.Generator) -> str:
    return CLEAR_PROMPTS[int(rng.integers(0, len(CLEAR_PROMPTS)))]


# ---------------------------------------------------------------------------
# 每集计划采样（四个随机的 ①②③）
# ---------------------------------------------------------------------------

def _weighted_sample_without_replacement(
    rng: np.random.Generator, weights: np.ndarray, k: int
) -> list[int]:
    """按权重不放回地抽 k 个下标(Gumbel top-k 技巧，等价于加权无放回抽样)。

    比"轮盘赌+重抽"稳定：不会在权重悬殊时反复抽到同一个下标。
    """
    n = len(weights)
    k = int(np.clip(k, 0, n))
    if k <= 0:
        return []
    w = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        w = np.ones(n, dtype=np.float64)
    gumbel = rng.gumbel(size=n)
    # 权重 0 的下标严格抽不到：+log(0) = -inf
    keys = np.where(w > 0, np.log(np.maximum(w, 1e-300)) + gumbel, -np.inf)
    return [int(i) for i in np.argsort(-keys)[:k]]


def _sample_episode_plan(
    rng: np.random.Generator,
    *,
    k_min: int,
    k_max: int,
    tool_weights: np.ndarray,
    plant_mode: str,
    fixed_order: bool,
    order_mode: str,
    k_weights: np.ndarray | None = None,
    plant_all_frac: float = 0.5,
) -> dict:
    """采样本集的 {plant_ids, grasp_ids, order}。

    - grasp_ids：要抓的那 k 把，按 tool_weights 加权无放回抽（螺丝刀权重更高 ⇒ 出现更频繁）。
    - plant_ids：场上种下的那些（grasp_ids 一定是子集，否则抓不到）。
        plant_mode="all"   → 永远 5 把全种，只抓其中 k 把；
        plant_mode="rand"  → 种 m 把，m ~ U[k, 5]，其余搬到机器人身后藏起来；
        plant_mode="mixed" → 按 plant_all_frac 先掷一次走 all 还是 rand。
        注意 rand 分支采到 m=5 时也会种满，所以 mixed 最终"场上 5 把"的占比
        明显高于 plant_all_frac（0.5 时实测 ≈0.80）。
    - order：grasp_ids 的排列。fixed_order 用 CLEAR_ORDER 里保序的子序列；
        order_mode="random" 均匀随机；"lr"/"rl" 固定从左到右/从右到左；
        "mixed" 在三者里等概率抽（默认 —— 保证既有"照视觉扫描"也有"随便抓"）。
    """
    ks = np.arange(int(k_min), int(k_max) + 1)
    if k_weights is not None and len(np.asarray(k_weights)) == len(ks):
        kw = np.asarray(k_weights, dtype=np.float64)
        kw = kw / kw.sum()
        k = int(rng.choice(ks, p=kw))
    else:
        k = int(rng.integers(int(k_min), int(k_max) + 1))
    grasp_ids = _weighted_sample_without_replacement(rng, tool_weights, k)

    # "mixed" 每集按 plant_all_frac 掷一次：一半匹配评测场景(永远 5 把都在)，
    # 一半只有 1..5 把 —— 前者保证不与评测分布错位，后者逼模型看"场上有什么"。
    if plant_mode == "mixed":
        plant_mode = "all" if rng.random() < float(plant_all_frac) else "rand"

    if plant_mode == "all":
        plant_ids = list(range(5))
    else:
        m = int(rng.integers(k, 5 + 1))
        # 场上必须有要抓的那几把，其余从剩下的里补
        rest = [t for t in range(5) if t not in grasp_ids]
        extra = _weighted_sample_without_replacement(
            rng, np.asarray(tool_weights, dtype=np.float64)[rest] if rest else np.zeros(0), m - k
        )
        plant_ids = sorted(grasp_ids + [rest[j] for j in extra])

    if fixed_order:
        order = [t for t in CLEAR_ORDER if t in grasp_ids]
    else:
        mode = order_mode
        if mode == "mixed":
            mode = str(rng.choice(["random", "lr", "rl"], p=[0.6, 0.2, 0.2]))
        if mode == "lr":
            order = sorted(grasp_ids)
        elif mode == "rl":
            order = sorted(grasp_ids, reverse=True)
        else:
            order = list(grasp_ids)
            rng.shuffle(order)

    return {"k": k, "grasp_ids": list(grasp_ids), "plant_ids": plant_ids, "order": order}


def _park_unplanted(env, base_body: str, plant_ids: list[int]) -> None:
    """把本集不种的工具搬到机器人身后 1.2m 外的地面上藏起来。

    必须"搬走"而不是"不种"：工具是 free joint，不写 qpos 就停在**上一集的落点**
    （大概率在箱子里或已入箱），既污染视觉又让布局校验误判。

    z 用该工具自己的参考高度（=桌面高）而不是负值：负值会掉到地面以下
    一直自由落体。放桌面高度则只是落到地板上停住，身后 1.2m 在相机视野外。
    """
    missing = [t for t in range(5) if t not in set(plant_ids)]
    if not missing:
        return
    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_rot = R.from_quat(np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)[[1, 2, 3, 0]])
    target_qpos = {}
    for tool_idx in missing:
        # 往 base 系 -x(机器人背后)方向推开，各自错开避免叠在一起
        pos_b = np.array(
            [-1.2 - 0.3 * tool_idx, 0.0, float(TOOL_REF_POS_B[tool_idx, 2])],
            dtype=np.float64,
        )
        world_pos = base_pos + base_rot.apply(pos_b)
        target_qpos[TOOL_JOINTS[tool_idx]] = np.concatenate(
            [world_pos, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)]
        )
    env.set_joint_qpos(target_qpos)
    env.mj_forward()


# ---------------------------------------------------------------------------
# 抓取守护(常开) + 抓取诊断(--grasp_probe 可选)：
#   守护：每把"S5-抬升"结束时校验工具是否真被抓起(抬离槽位)。若空抓(工具仍躺槽位)，
#         立即中止本集——省掉"空抓后仍白飞一趟箱子、直到放置检测才判负"的墙钟(每败省~2min)。
#         空抓本来 100% 会在放置检测被 world 接触判负，此处只是提前失败、不改成功路径。
#   诊断：在"S4-沉降闭爪"结束 与 "S5-抬升"结束打点，打印 指令/末端/工具实测，
#         事后判断空抓是几何错位还是跟踪误差。
# ---------------------------------------------------------------------------

_LIFT_RISE_MIN = 0.10  # 抬升校验阈值(m)：抬后 tool body z 需 ≥ 槽位z + 0.10 才算抓起


class _LiftGuardDevice(_T.G1ScriptedTrajectoryDevice):
    """常开设备：S5 抬升后若工具未离槽即判空抓并中止；可选抓取诊断打点。"""

    def __init__(
        self, *a, lift_check_at=None, probe_at=None,
        settle_close_at=None, g_open=None, step_tool=None,
        tool_id_holder=None, **k,
    ):
        super().__init__(*a, **k)
        self.lift_check_at = dict(lift_check_at or {})
        self.probe_at = dict(probe_at or {})
        # 逐控制步的工具标签(0..4；-1=撤离/等待)。每个 update() 把"当前步"的标签
        # 发布进 tool_id_holder，采集端读它写进数据集的 tool_id 列。
        # 数据集那一列自带一帧滞后(见 collection_data)，正好对齐 state/images。
        self.step_tool = None if step_tool is None else np.asarray(step_tool, dtype=np.int64)
        self.tool_id_holder = tool_id_holder
        # settle_close_at: {轨迹步(S4起点): (容差m, 最多额外等待步)}
        # 末端未到达该位姿前不放行闭爪，消除时序性空抓/贴边咬合。
        self.settle_close_at = dict(settle_close_at or {})
        self.g_open = float(g_open) if g_open is not None else None
        self._settle_held: dict[int, int] = {}

    def _cmd_hold_open(self):
        """保持当前目标位姿、夹爪张开（不推进 t）。"""
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        l_n = len(self.l_grip.ctrl_index)
        r_n = len(self.r_grip.ctrl_index)
        if self.g_open is not None:
            self.l_grip.update_ctrl(np.full(l_n, self.g_open, dtype=np.float32))
            self.r_grip.update_ctrl(np.full(r_n, self.g_open, dtype=np.float32))

    def _query_tool_pos(self, tool_idx: int) -> np.ndarray:
        return np.asarray(
            self.r_arm.env.query_position_body_B(
                TOOL_JOINTS[tool_idx], self.r_arm.base_link
            ),
            dtype=np.float64,
        )

    def update(self):
        # 发布本控制步的工具标签(在一切早退之前，保证每一步都被记到)
        if self.step_tool is not None and self.tool_id_holder is not None:
            idx = min(int(self.t), len(self.step_tool) - 1)
            self.tool_id_holder["v"] = int(self.step_tool[idx])
        # ── 沉降到位再闭爪：S4 起点若末端未接近指令位姿，张开保持等待，到位才放行 ──
        if not self.place_failed and self.t in self.settle_close_at and self.g_open is not None:
            tol, max_extra, ti = self.settle_close_at[self.t]
            held = self._settle_held.get(self.t, 0)
            converged = False
            err = None
            try:
                ee_b = self.r_arm.env.query_site_pos_and_quat_B(
                    [self.r_arm.ee_name], [self.r_arm.base_link]
                )
                ee = np.asarray(ee_b[self.r_arm.ee_name]["xpos"], dtype=np.float64)
                err = float(
                    np.linalg.norm(ee - np.asarray(self.r_pos[self.t], dtype=np.float64))
                )
                converged = err <= tol
            except Exception:
                converged = True  # 查询失败不阻塞流程
            if (not converged) and held < int(max_extra):
                self._settle_held[self.t] = held + 1
                self._cmd_hold_open()
                return
            if held > 0:
                print(
                    f"  [settle] {TOOL_NAMES[ti]} 张开等待 {held} 步后放行闭爪"
                    f"(err={'NA' if err is None else round(err * 1000)}mm, t={self.t})",
                    flush=True,
                )
        if self.t in self.lift_check_at:
            try:
                ti, strip_z = self.lift_check_at[self.t]
                tb = self._query_tool_pos(ti)
                z = float(tb[2])
                if z < strip_z + _LIFT_RISE_MIN:
                    name = TOOL_NAMES[ti]
                    self.place_failed = True
                    self.place_fail_reason = (
                        f"抬升校验: 工具 {name} 未随末端抬起"
                        f"(槽位z={strip_z:.3f}, 抬后z={z:.3f})"
                    )
                    print(
                        f"  ✗ 抬升校验未通过: {name} 未离槽(z={z:.3f} < "
                        f"{strip_z + _LIFT_RISE_MIN:.3f}) → 立即中止本集", flush=True,
                    )
                    orca_logger.warning(f"[抬升校验] {self.place_fail_reason}")
                    self._force_task_end()
                    return
            except Exception as e:
                print(f"[抬升校验] 查询失败 {type(e).__name__}: {e}", flush=True)
        if self.t in self.probe_at:
            try:
                ti, when = self.probe_at[self.t]
                name = TOOL_NAMES[ti]
                cmd = (
                    np.asarray(self.r_pos[self.t], dtype=np.float64)
                    if self.t < len(self.r_pos)
                    else None
                )
                ee_b = self.r_arm.env.query_site_pos_and_quat_B(
                    [self.r_arm.ee_name], [self.r_arm.base_link]
                )
                ee = np.asarray(ee_b[self.r_arm.ee_name]["xpos"], dtype=np.float64)
                ee_q = np.asarray(ee_b[self.r_arm.ee_name]["xquat"], dtype=np.float64)
                tb = self._query_tool_pos(ti)
                tq = None
                try:
                    tq = _T._body_rot_in_B(
                        self.r_arm.env, TOOL_JOINTS[ti], self.r_arm.base_link
                    ).as_quat()  # xyzw → 转 wxyz
                    tq = np.asarray(tq[[3, 0, 1, 2]], dtype=np.float64)
                except Exception:
                    pass
                print(
                    f"[GRASP-PROBE] {when} {name} t={self.t} "
                    f"cmd={None if cmd is None else np.round(cmd, 3).tolist()} "
                    f"ee={np.round(ee, 3).tolist()} tool={np.round(tb, 3).tolist()} "
                    f"ee_q={np.round(ee_q, 3).tolist()} "
                    f"tq={'NA' if tq is None else np.round(tq, 3).tolist()}",
                    flush=True,
                )
            except Exception as e:
                print(f"[GRASP-PROBE] err {type(e).__name__}: {e}", flush=True)
        super().update()


def _tool_of_label(lbl: str) -> int | None:
    """从 "收2/5-螺丝刀#1-S4-沉降闭爪" 解出真实工具索引(1)；非収段返回 None。

    **不能**用 "收N/" 里的 N 反推工具索引 —— N 是抓取序号，顺序随机后与
    工具索引不再相等。工具索引只认 `#N`。
    """
    if not lbl.startswith("收") or "#" not in lbl:
        return None
    try:
        return int(lbl.split("#", 1)[1].split("-", 1)[0])
    except (IndexError, ValueError):
        return None


def _build_guard_steps(
    all_segments, strip_z_by_tool=None, settle_cfg=None
) -> tuple[dict[int, tuple[int, str]], dict[int, tuple[int, float]], dict]:
    """扫描全轨迹段，返回 (probe_at, lift_at, settle_at)。

    probe_at: {轨迹步: (tool_idx, 时机)}，时机 ∈ grasp(闭爪后) / lifted(抬升后)。
    lift_at:  {轨迹步: (tool_idx, 槽位z)}，仅登记每把"S5-抬升"结束步。
    settle_at:{轨迹步(S4起点): (容差m, 最多额外等待步, tool_idx)}，
              settle_cfg=(tol,max_extra)。
    """
    probe_at: dict[int, tuple[int, str]] = {}
    lift_at: dict[int, tuple[int, float]] = {}
    settle_at: dict[int, tuple[float, int, int]] = {}
    settle_on = settle_cfg is not None and float(settle_cfg[0]) > 0
    cum = 0
    for s in all_segments:
        lbl = str(s.get("label", ""))
        steps = int(s["steps"])
        if lbl.startswith("收") and ("S4-沉降闭爪" in lbl):
            ti = _tool_of_label(lbl)
            if ti is None:
                cum += steps
                continue
            probe_at[cum] = (ti, "s4start")
            probe_at[cum + steps - 1] = (ti, "grasp")
            if settle_on:
                settle_at[cum] = (
                    float(settle_cfg[0]), int(max(0, settle_cfg[1])), ti,
                )
        elif lbl.startswith("收") and ("S5-抬升" in lbl):
            ti = _tool_of_label(lbl)
            if ti is None:
                cum += steps
                continue
            probe_at[cum + steps - 1] = (ti, "lifted")
            if strip_z_by_tool is not None and ti in strip_z_by_tool:
                lift_at[cum + steps - 1] = (ti, float(strip_z_by_tool[ti]))
        cum += steps
    return probe_at, lift_at, settle_at


def _build_step_tool(all_segments) -> np.ndarray:
    """逐 env 控制步的工具标签(0..4)，撤离/入箱等待段 = -1。

    直接从段 label 前缀解析，与 _build_guard_steps 共用同一套约定，
    避免"哪一步属于哪把工具"在两处各记一份而漂移。
    """
    out: list[int] = []
    for s in all_segments:
        ti = _tool_of_label(str(s.get("label", "")))
        out.extend([-1 if ti is None else ti] * int(s["steps"]))
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# 布局采样与种植（标称槽位 ± 小抖动；低频"被碰歪"）
# ---------------------------------------------------------------------------

def _sample_clear_layout(
    rng: np.random.Generator,
    j_x=0.008, j_y=0.018, j_yaw_deg=6.0,
    big_frac=0.12, big_y=0.040, big_yaw_deg=14.0,
    min_gap=0.045, tries=80,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """返回 (xy, yaw) 两个 dict，键=工具索引 0..4。

    xy 是 base 系条带上的种植中心(x,y)，yaw 是绕 base 竖直轴的小旋转(rad)。
    所有扰动都让工具仍"齐整地"留在条带内：垂直带 x 只微动、沿带 y 小抖、
    绕竖轴小幅旋转；低频(~12%)把随机一把再加"碰歪"(y 与 yaw 幅度更大)。
    两两中心距离 ≥ min_gap，避免抖出重叠。
    """
    ref = TOOL_REF_POS_B
    jyaw = float(np.deg2rad(j_yaw_deg))
    byaw = float(np.deg2rad(big_yaw_deg))

    for _ in range(tries):
        # 每集随机选要不要"碰歪"一把（先定，避免多把同时大偏）
        big_idx = None
        if rng.random() < big_frac:
            big_idx = int(rng.integers(0, 5))

        xy: dict[int, tuple[float, float]] = {}
        yaw: dict[int, float] = {}
        ok = True
        for i in range(5):
            big = (i == big_idx)
            if big:
                dx = float(rng.uniform(-j_x, j_x))
                dy = float(rng.uniform(-big_y, big_y))
                dz_rot = float(rng.uniform(-byaw, byaw))
            else:
                dx = float(rng.uniform(-j_x, j_x))
                dy = float(rng.uniform(-j_y, j_y))
                dz_rot = float(rng.uniform(-jyaw, jyaw))
            xy[i] = (round(float(ref[i, 0]) + dx, 4), round(float(ref[i, 1]) + dy, 4))
            yaw[i] = dz_rot

        # 两两中心距离检查
        cents = np.array([[xy[i][0], xy[i][1]] for i in range(5)], dtype=np.float64)
        d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= min_gap:
            return xy, yaw
    # 实在采不到合法布局 → 返回纯标称(零抖动)，尽量保证一次能落地
    orca_logger.warning("[布局] 抖动采样多次冲突，退回纯标称布局")
    xy = {i: (float(ref[i, 0]), float(ref[i, 1])) for i in range(5)}
    return xy, {i: 0.0 for i in range(5)}


def _plant_tools(
    env, base_body: str,
    xy: dict[int, tuple[float, float]],
    yaw: dict[int, float],
    plant_ids=None,
) -> None:
    """把 plant_ids 里的工具种回 (标称+抖动) 位姿：自由关节 qpos 直接写世界位姿。

    plant_ids=None 等价于全部 5 把(旧行为)。不在名单里的由 _park_unplanted 藏起来。
    """
    ids = list(range(5)) if plant_ids is None else list(plant_ids)
    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    target_qpos = {}
    for tool_idx in ids:
        x, y = xy[tool_idx]
        pos_b = np.array([x, y, float(TOOL_REF_POS_B[tool_idx, 2])], dtype=np.float64)
        # 标称姿态再绕 base 竖直轴转 yaw(先乘 = 在 base 系绕 z)
        rot_ref = R.from_quat(np.asarray(TOOL_REF_QUAT_XYZW_B[tool_idx], dtype=np.float64))
        rot_b = R.from_euler("z", float(yaw[tool_idx])) * rot_ref
        world_pos = base_pos + base_rot.apply(pos_b)
        world_quat_xyzw = (base_rot * rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        target_qpos[TOOL_JOINTS[tool_idx]] = np.concatenate([world_pos, world_quat_wxyz])
    env.set_joint_qpos(target_qpos)
    env.mj_forward()


def _settle(env, mgr, steps: int) -> None:
    if steps <= 0:
        return
    try:
        mgr.set_init_ctrl()
        env.set_ctrl(mgr.ctrl)
        for _ in range(int(steps)):
            env.step(mgr.ctrl)
    except Exception as e:
        orca_logger.warning(f"[settle] 物理沉降异常: {e}")


def _query_pos_b(env, base_body: str) -> np.ndarray:
    return _T._query_tool_positions_b(env, base_body)


def _query_quat_b(env, base_body: str) -> np.ndarray:
    """5 件工具 body 相对 base 系的姿态 (5,4) xyzw —— 刚体形变要用。"""
    return _T._query_tool_quats_b(env, base_body)


def _validate_layout(
    actual: np.ndarray, z_lo=0.105, z_hi=0.22, min_gap=0.05, ids=None,
) -> tuple[bool, str]:
    """静置后校验：**场上**那几把都在桌面高度、没落桌下/弹飞、两两有安全间距。

    ids=None 等价于全部 5 把。--plant_mode rand 时被藏到桌下的那些**必须排除**，
    否则永远判 z<z_lo 失败。
    """
    sel = actual if ids is None else actual[np.asarray(list(ids), dtype=np.int64)]
    if len(sel) == 0:
        return True, "(空场)"
    z = sel[:, 2]
    if any(z < z_lo) or any(z > z_hi):
        return False, f"静置高度越界 {np.round(z, 3).tolist()}"
    cents = sel[:, :2]
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if d.min() < min_gap:
        return False, f"实际间距过近 min={d.min():.3f}"
    return True, f"z∈[{np.round(z.min(), 3)},{np.round(z.max(), 3)}], min_gap={d.min():.3f}"


# ---------------------------------------------------------------------------
# 放箱错位（防 5 把堆同一中心）
# ---------------------------------------------------------------------------

# 这些段的 r_target_b 属于"放箱插入"轨迹，需要整体平移 xy 来错开落点。
_BOX_SEG_MARKERS = ("移到箱上", "逼近松开位", "沉降松开", "松开后抬升")


def _offset_box_segments(segs: list[dict], ox: float, oy: float) -> list[dict]:
    """把属于放箱插入的 segment 的 xy 目标平移 (ox, oy)，返回新列表(不改原)。"""
    out = []
    for s in segs:
        label = str(s.get("label", ""))
        if any(m in label for m in _BOX_SEG_MARKERS):
            s = dict(s)
            pos = list(s["r_target_b"])
            pos[0] = float(pos[0]) + float(ox)
            pos[1] = float(pos[1]) + float(oy)
            s["r_target_b"] = pos
        out.append(s)
    return out


def _sample_box_drop_offsets(
    box_bottom: dict,
    rng: np.random.Generator,
    anchor: np.ndarray,
    n: int = 5,
    spread_xy: float = 0.10,
    min_sep: float = 0.05,
) -> np.ndarray:
    """在箱底内侧标定矩形内、锚点周围小邻域给 n 把工具采错位落点 offsets，shape (n,2)。

    offsets 都是**相对锚点**的偏移：offsets[0]=(0,0)，第 1 把放锚点(即模板放箱点，
    与既往 single_place 的放箱点一致)；其余在 [anchor−spread_xy, anchor+spread_xy]
    内、且钳到箱底内面(避开箱壁)取与已有点 ≥ min_sep 的点，最后整体相对锚点返回。
    邻域有限(spread_xy 默认 0.10)既保证落点仍在箱口上方、OSC 可达，又把 5 把的
    堆叠位置分开，避免全砸同一中心把堆顶拱到轨迹上。
    """
    anchor = np.asarray(anchor, dtype=np.float64).reshape(2)
    try:
        xy_min = np.asarray(box_bottom["xy_min"], dtype=np.float64)
        xy_max = np.asarray(box_bottom["xy_max"], dtype=np.float64)
    except Exception as e:
        orca_logger.warning(f"[放箱] 箱底几何不可用，使用中心落点: {e}")
        return np.zeros((n, 2), dtype=np.float64)

    lo = np.maximum(anchor - spread_xy, xy_min + 0.02)
    hi = np.minimum(anchor + spread_xy, xy_max - 0.02)
    if hi[0] - lo[0] < 0.02 or hi[1] - lo[1] < 0.02:
        orca_logger.warning(
            f"[放箱] 锚点邻域偏小 x∈[{lo[0]:.2f},{hi[0]:.2f}] y∈[{lo[1]:.2f},{hi[1]:.2f}]，退回中心落点"
        )
        return np.zeros((n, 2), dtype=np.float64)

    pts = [anchor.copy()]
    for _ in range(n - 1):
        for _try in range(400):
            q = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])], dtype=np.float64)
            if all(np.linalg.norm(q - p) >= min_sep for p in pts):
                pts.append(q)
                break
        else:
            # 采不到足够分开的点就补一个邻域内的点(仍相对锚点错开)
            q = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])], dtype=np.float64)
            pts.append(q)
    pts = np.stack(pts, axis=0)     # (n,2) 绝对落点
    return pts - anchor              # 相对锚点偏移(第 1 把 = 0,0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="G1 Clear-All(随机顺序·随机把数·不点名) 脚本化采集 → LeRobot v2.1"
    )
    ap.add_argument("--task_config", type=str, default="../common/example.yaml")
    ap.add_argument(
        "--lerobot_out",
        # 新数据集，**不要**写回 g1_omnipicker_clear5 —— 那是跑通了基线的 181 集固定顺序集
        type=str,
        default=os.path.join(OPENPI_DATA_ROOT, "g1_omnipicker_clear5_rand"),
    )
    ap.add_argument("--repo_id", default="local/g1_omnipicker_clear5_rand")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    ap.add_argument("--num_episodes", type=int, default=40,
                    help="数据集目标总集数(含已存)；配合 --resume 可分块续采")
    ap.add_argument("--max_attempts", type=int, default=8, help="单集成功前的最大尝试次数")
    ap.add_argument("--grasp_probe", action="store_true", help="逐把闭爪/抬升后打印 指令vs末端vs工具 (诊断空抓)")
    ap.add_argument("--no_lift_check", action="store_true",
                    help="关闭 S5 抬升后的空抓早中止(默认开，空抓立即中止省墙钟)")
    # 抓取修复：指令 z 压到末端可达下限之下 → OSC 保持残余误差=持续向下座椅力
    ap.add_argument("--grasp_dz_mm", type=float, default=-12.0,
                    help="抓取指令 z 补偿(mm，负=降)。探针证据:末端浮高 ~0.168 是物理下限、降不到更低，"
                         "但把指令压到下限之下会给 OSC 持续下压力，闭爪时把薄工具'坐'进钳口(座椅力)。"
                         "dz=0(目标可达)时静悬闭爪抓不牢薄扳手。默认 -12。")
    ap.add_argument("--settle_tol_mm", type=float, default=8.0,
                    help="闭爪前末端距指令位姿的收敛容差(mm)；0=关闭沉降门")
    ap.add_argument("--settle_max_extra", type=int, default=250,
                    help="沉降门最多额外等待的控制步数(~0.005s/步)")
    ap.add_argument("--warp-rigid", action=argparse.BooleanOptionalAction, default=False,
                    help="抓取路点按工具实测**位姿**做刚体形变(默认关)。开=多绕工具原点转 dR 那一步，"
                         "关=旧的纯平移。动机：工具被随机化 yaw(±6°，12%%概率碰歪±14°)，"
                         "而模板抓取点是按标称姿态标定的，平移形变不跟着转。"
                         "**尚未证明有效**：yaw 6° 在 80mm 杠杆臂上只有 ~8mm，而实测空抓的"
                         "差异主要在 z(22mm)，yaw 解释不了 —— 开着只为做 A/B。")
    ap.add_argument("--safe_z", type=float, default=0.48)
    ap.add_argument("--settle_steps", type=int, default=180)
    ap.add_argument("--kp", type=float, default=220.0)
    # 布局抖动
    ap.add_argument("--j_x", type=float, default=0.008, help="垂直带方向 x 抖动(±m)")
    ap.add_argument("--j_y", type=float, default=0.018, help="沿带 y 抖动(±m)")
    ap.add_argument("--j_yaw_deg", type=float, default=6.0, help="绕竖轴抖动(±°)")
    ap.add_argument("--big_frac", type=float, default=0.12, help="每集抽一把再'碰歪'的概率")
    ap.add_argument("--big_y", type=float, default=0.040, help="碰歪那把的 y 幅度(±m)")
    ap.add_argument("--big_yaw_deg", type=float, default=14.0, help="碰歪那把的倾角幅度(±°)")
    # 放箱错位
    ap.add_argument("--no_box_spread", action="store_true",
                    help="关闭放箱 xy 错位(全部落同一中心)")
    ap.add_argument(
        "--smoke", type=int, default=0,
        help=">0: 冒烟模式，每集把 head/wrist 首末帧存到 /tmp/clear5_debug/，共 smoke 集",
    )
    ap.add_argument(
        "--waypoint_files",
        type=str,
        default=",".join(os.path.join(base_dir, f"my_waypoint_tool{i}.yaml") for i in range(1, 6)),
    )
    ap.add_argument(
        "--place_check_timeout_s", type=float, default=_PLACE_CHECK_TIMEOUT_S,
        help="EE 出整箱矩形后等待有效放置接触的最大秒数",
    )
    # 各分段帧数(采集帧，运行时换算控制步)
    ap.add_argument("--f_transit", type=int, default=50)
    ap.add_argument("--f_descend", type=int, default=30)
    ap.add_argument("--f_grasp", type=int, default=35)
    ap.add_argument("--f_settle", type=int, default=30)
    ap.add_argument("--f_lift", type=int, default=35)
    ap.add_argument("--f_place_via", type=int, default=45)
    ap.add_argument("--f_to_box", type=int, default=50)
    ap.add_argument("--f_release", type=int, default=45)
    ap.add_argument("--f_release_settle", type=int, default=20)
    ap.add_argument("--f_lift_after", type=int, default=15)
    ap.add_argument("--f_retreat", type=int, default=_PLACE_RETREAT_STEPS)

    # ── 随机化(2026-09-23)：①②③④ ──────────────────────
    ap.add_argument("--k_min", type=int, default=1, help="①每集最少抓几把")
    ap.add_argument("--k_max", type=int, default=5, help="①每集最多抓几把")
    ap.add_argument("--k_weights", type=str, default="1,1,2,3,4",
                    help="k=k_min..k_max 的抽样权重(逗号分隔)。默认向大 k 倾斜(评测是 5 把)")
    ap.add_argument("--tool_weights", type=str, default="1,3,1,1,1",
                    help="5 把工具被抽中抓取的权重(逗号分隔，顺序同 TOOL_NAMES)。"
                         "默认螺丝刀(index 1)=3，满足'螺丝刀多一些数据量'")
    ap.add_argument("--plant_mode", choices=["all", "rand", "mixed"], default="mixed",
                    help="③all=永远种 5 把；rand=种 m∈[k,5] 把，其余搬到身后；"
                         "mixed(默认)=每集按 --plant_all_frac 掷一次")
    ap.add_argument("--plant_all_frac", type=float, default=0.5,
                    help="--plant_mode mixed 时走 all 分支的概率。注意实测最终'种满 5 把'"
                         "的集占比 ≈ 0.80 而非 0.50 —— rand 分支采到 m=5 也会种满")
    ap.add_argument("--order_mode", choices=["mixed", "random", "lr", "rl"], default="mixed",
                    help="抓取顺序：mixed=60%%随机/20%%从左到右/20%%从右到左")
    ap.add_argument("--fixed_order", action="store_true",
                    help="退回固定 CLEAR_ORDER(从左到右)顺序")
    ap.add_argument("--init_jitter_deg", type=float, default=6.0,
                    help="④双臂初始关节角抖动(±°，0=关)")
    args = ap.parse_args()

    tool_weights = np.asarray(
        [float(v) for v in args.tool_weights.split(",")], dtype=np.float64
    )
    if len(tool_weights) != 5:
        orca_logger.error(f"--tool_weights 需要 5 个数，收到 {len(tool_weights)}")
        return
    if args.k_min > args.k_max:
        orca_logger.error(f"--k_min({args.k_min}) > --k_max({args.k_max})")
        return
    k_weights = np.asarray([float(v) for v in args.k_weights.split(",")], dtype=np.float64)
    if len(k_weights) != args.k_max - args.k_min + 1:
        orca_logger.warning(
            f"--k_weights 长度 {len(k_weights)} 与 k∈[{args.k_min},{args.k_max}] 不符，改用均匀"
        )
        k_weights = None

    if args.big_y <= args.j_y:
        orca_logger.warning("[布局] big_y 应大于 j_y 才是'碰歪'，已自动调大")
        args.big_y = args.j_y * 2.0

    # ── 路点 ─────────────────────────────────────────────
    tool_data: list[dict] = []
    g_open_global = -0.8561
    g_close_global = 2.0
    for i, path in enumerate([p.strip() for p in args.waypoint_files.split(",")]):
        if not os.path.exists(path):
            orca_logger.error(f"路点文件不存在: {path}")
            return
        td = _T._load_waypoint_yaml(os.path.abspath(path))
        tool_data.append(td)
        if i == 0:
            g_open_global = td["g_open"]
            g_close_global = td["g_close"]

    # ── 环境/控制器/相机 ─────────────────────────────────
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    _n_motor = len(agent_conf.gripper_l["actuator_names"]) + len(agent_conf.gripper_r["actuator_names"])

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_clear5")
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    env_dt = float(manager.time_step) * int(manager.frame_skip)
    frame_dt = 1.0 / float(args.fps)
    steps_scale = max(1, int(round(frame_dt / env_dt)))
    scaled_dt = steps_scale * env_dt
    orca_logger.info(f"[时序] steps_scale={steps_scale} (env_dt={env_dt:.4f}, frame_dt={frame_dt:.4f})")
    if abs(scaled_dt - frame_dt) > 1e-6:
        orca_logger.warning("轨迹控制周期与采集帧率存在偏差，请检查时序")

    def _ctrl_steps(frames: int) -> int:
        return max(1, int(round(float(frames) * steps_scale)))

    print("=" * 68, flush=True)
    print("  G1 OmniPicker · Clear-All：随机顺序/随机把数 / 不点名", flush=True)
    print(f"  目标总集数 {args.num_episodes} | 抖动 y±{args.j_y} x±{args.j_x} "
          f"yaw±{args.j_yaw_deg}° | 碰歪 p={args.big_frac} (y±{args.big_y} yaw±{args.big_yaw_deg}°)",
          flush=True)
    print(f"  每集抓 k∈[{args.k_min},{args.k_max}] 权重 {args.k_weights} | "
          f"场上 {args.plant_mode} | 顺序 {args.order_mode}"
          f"{' (固定)' if args.fixed_order else ''} | 初始抖动 ±{args.init_jitter_deg}°", flush=True)
    print(f"  抓取权重 [{' '.join(f'{TOOL_NAMES[i]}:{tool_weights[i]:g}' for i in range(5))}]",
          flush=True)
    print(f"  放箱错位 {'开' if not args.no_box_spread else '关'} | safe_z={args.safe_z}", flush=True)
    print(f"  输出: {args.lerobot_out}", flush=True)
    print("=" * 68, flush=True)

    # ── 相机与数据集 ─────────────────────────────────────
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = dict(DEFAULT_CAMERA_MAP)
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")
    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return
    cam_shape = (3, cam_hw[0], cam_hw[1])
    print(f"  相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}", flush=True)

    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=args.lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_omnipicker",
        extra_features={
            # 逐帧工具标签(0..4，-1=撤离/等待)。**标签不是输入**：评测时"这是哪把"
            # 正是模型要从视觉推的，喂进去等于给答案；它只用于分工具加权损失。
            "tool_id": {"dtype": "int64", "shape": (1,), "names": None},
        },
    )
    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task=CLEAR_PROMPTS[0],
        clock="sim",
    )

    # ── 控制器 ───────────────────────────────────────────
    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        orca_logger.error("场景初始化失败")
        env.close()
        return
    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])
    base_body = env.body(agent_conf.base_body)

    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)
    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)

    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # 工具箱底/整箱几何标定一次(位置基座系固定)。
    box_bottom = None
    try:
        box_bottom = _T._calibrate_box_bottom_inner(env, base_body)
        _xy_min = np.asarray(box_bottom.get("box_xy_min", box_bottom["xy_min"]))
        _xy_max = np.asarray(box_bottom.get("box_xy_max", box_bottom["xy_max"]))
        _in_min = np.asarray(box_bottom["xy_min"])
        _in_max = np.asarray(box_bottom["xy_max"])
        orca_logger.info(
            f"[放置检测] 箱底标定完成: 内面 {_in_min.round(3).tolist()}~{_in_max.round(3).tolist()}, "
            f"整箱 {_xy_min.round(3).tolist()}~{_xy_max.round(3).tolist()}, "
            f"z_surface={float(box_bottom['z_surface']):.3f}"
        )
        print(f"  箱底内面 {_in_min.round(3).tolist()}~{_in_max.round(3).tolist()} "
              f"(整箱 {_xy_min.round(3).tolist()}~{_xy_max.round(3).tolist()})", flush=True)
    except Exception as e:
        orca_logger.error(f"[放置检测] 工具箱底标定失败: {e}")
        env.close()
        return

    rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))

    n_target = int(args.num_episodes)
    n_success_run = 0
    stats = {
        "attempt": 0, "success": 0,
        "fail": {"place": 0, "layout": 0, "other": 0, "giveup": 0},
    }
    smoke_budget = max(0, int(args.smoke))
    if smoke_budget:
        os.makedirs("/tmp/clear5_debug", exist_ok=True)
    else:
        smoke_budget = -1

    # 每把工具的一次完整 pick+place = 单把帧数(帧)；最终每集 ≈ 5×单把 + 末段撤离+检查
    per_tool_frames = sum([
        args.f_transit, args.f_descend, args.f_grasp, args.f_settle,
        args.f_lift, 2 * args.f_place_via, args.f_to_box, args.f_release,
        args.f_release_settle, args.f_lift_after,
    ])
    print(f"  单把约 {per_tool_frames} 帧 × 5 + 撤离 ≈ {5 * per_tool_frames + args.f_retreat + 60} 帧/集\n", flush=True)

    def _dump_frames(tag: str, ep_display: int):
        try:
            from dataStorage.lerobot_camera import capture_frame_with_idx
            imgs, _ = capture_frame_with_idx(cameras, camera_map, cam_hw)
            for k, arr in imgs.items():
                from PIL import Image
                Image.fromarray(arr).save(f"/tmp/clear5_debug/ep{ep_display}_{tag}_{k}.png")
        except Exception as e:
            orca_logger.warning(f"[冒烟] 保存调试帧失败: {e}")

    try:
        with writer:
            attempts_since_save = 0
            layout_fail_streak = 0
            giveup_streak = 0
            while writer.num_episodes < n_target:
                attempts_since_save += 1
                if attempts_since_save > args.max_attempts:
                    stats["fail"]["giveup"] += 1
                    giveup_streak += 1
                    attempts_since_save = 0
                    storage.clear_data()
                    orca_logger.warning(
                        f"[放弃] 连续 {args.max_attempts} 次未保存，重开新一轮"
                        f"(已存 {writer.num_episodes})"
                    )
                    print(f"  [放弃] 连续 {args.max_attempts} 次失败，跳到下一轮", flush=True)
                    if giveup_streak >= 30:
                        orca_logger.error(
                            "[放弃] 连续 30 轮未能保存任何一集，疑似场景/轨迹系统性不匹配，停止"
                        )
                        print(">>> [放弃] 连续 30 轮未保存，停止采集", flush=True)
                        break
                    continue
                stats["attempt"] += 1
                attempt_no = stats["attempt"]
                ep_display = writer.num_episodes + 1

                plan = _sample_episode_plan(
                    rng,
                    k_min=args.k_min, k_max=args.k_max,
                    tool_weights=tool_weights,
                    plant_mode=args.plant_mode,
                    fixed_order=args.fixed_order,
                    order_mode=args.order_mode,
                    k_weights=k_weights,
                    plant_all_frac=args.plant_all_frac,
                )
                order = plan["order"]
                # ④ 起始位姿：双肩/肘关节加小扰动。必须在 build_segmented_trajectory
                # 之前生效 —— l_hold/r_hold 段是"保持建轨迹那一刻的实测位姿"，
                # 先抖动后建轨迹才不会在开头出现一帧跳变。
                ep_joint_values = dict(default_joint_values)
                if args.init_jitter_deg > 0:
                    jit = float(np.deg2rad(args.init_jitter_deg))
                    for jn in list(agent_conf.l_arm["joint_names"]) + list(
                        agent_conf.r_arm["joint_names"]
                    ):
                        ep_joint_values[jn] = float(
                            np.asarray(default_joint_values[jn]).reshape(-1)[0]
                            + rng.uniform(-jit, jit)
                        )
                print(
                    f"  [计划] 场上 {len(plan['plant_ids'])} 把 "
                    f"{[TOOL_NAMES[t] for t in plan['plant_ids']]}"
                    f" | 抓 {len(order)} 把: {' → '.join(TOOL_NAMES[t] for t in order)}",
                    flush=True,
                )

                xy, yaw = _sample_clear_layout(
                    rng, j_x=args.j_x, j_y=args.j_y, j_yaw_deg=args.j_yaw_deg,
                    big_frac=args.big_frac, big_y=args.big_y, big_yaw_deg=args.big_yaw_deg,
                )

                env.reset()
                time.sleep(0.05)
                if not manager.update_scene():
                    orca_logger.info("场景更新失败，停止采集")
                    break
                env.set_default_joint_values(ep_joint_values)
                base_body = env.body(agent_conf.base_body)

                _plant_tools(env, base_body, xy, yaw, plant_ids=plan["plant_ids"])
                _park_unplanted(env, base_body, plan["plant_ids"])
                _settle(env, manager, args.settle_steps)
                env.set_default_joint_values(ep_joint_values)  # 重新归位机器人
                env.mj_forward()
                base_body = env.body(agent_conf.base_body)

                actual = _query_pos_b(env, base_body)
                # 姿态只在 --warp-rigid 下才用得上；默认(关)不查，省一次 gRPC 且
                # 避免这条没人走的路径悄悄烂掉（2026-09-23 就是这样崩过一次）。
                actual_quat = _query_quat_b(env, base_body) if args.warp_rigid else None
                good, why = _validate_layout(actual, ids=plan["plant_ids"])
                if not good:
                    stats["fail"]["layout"] += 1
                    layout_fail_streak += 1
                    orca_logger.warning(f"[布局] 校验未通过: {why}，重采")
                    print(f"  [布局] 校验未通过: {why}，重采", flush=True)
                    if layout_fail_streak >= 30:
                        orca_logger.error(
                            "[布局] 连续 30 次校验失败，疑似抖动/沉降参数与场景不符，停止"
                        )
                        print(">>> [布局] 连续 30 次失败，停止采集", flush=True)
                        break
                    continue
                layout_fail_streak = 0
                print(f"  [布局] ok {why}", flush=True)

                prompt = _clear_prompt(rng)
                storage.set_task(prompt)

                # 放箱错位偏移(第1把 = 模板放箱点正中，其余邻域内错开)
                # 锚点 = 模板放箱路点(所有工具的 yaml 放箱点一致)
                box_anchor = np.asarray(tool_data[0]["waypoints"][-1]["pos"], dtype=np.float64)[:2]
                spread_on = not args.no_box_spread
                drop_off = _sample_box_drop_offsets(box_bottom, rng, box_anchor, n=5) \
                    if spread_on else np.zeros((5, 2), dtype=np.float64)
                if spread_on:
                    print(f"  [放箱] 锚点 {np.round(box_anchor, 3).tolist()} "
                          f"错位偏移 {np.round(drop_off, 3).tolist()}", flush=True)

                place_timeout_steps = max(
                    1, int(round(float(args.place_check_timeout_s) * args.fps)) * steps_scale,
                )

                # ── 拼接 5 段 pick+place 轨迹 ─────────────────────
                all_segments: list[dict] = []
                place_arm_at: dict[int, int] = {}
                cum_steps = 0
                last_release_quat = None
                layout_txt = []
                for order_no, tool_idx in enumerate(order):
                    x, y = xy[tool_idx]
                    layout_txt.append(f"{order_no + 1}.{TOOL_NAMES[tool_idx]}@({x:.3f},{y:.3f})")
                    delta_b = actual[tool_idx] - TOOL_REF_POS_B[tool_idx]
                    if args.warp_rigid:
                        # 工具的姿态也被随机化了(yaw ±6°，12% 概率碰歪 ±14°)，
                        # 只平移会让抓取点绕工具原点画弧 ⇒ 空抓。见 _warp_waypoints_rigid。
                        warped = _T._warp_waypoints_rigid(
                            tool_data[tool_idx],
                            ref_pos=TOOL_REF_POS_B[tool_idx],
                            ref_quat=TOOL_REF_QUAT_XYZW_B[tool_idx],
                            cur_pos=actual[tool_idx],
                            cur_quat=actual_quat[tool_idx],
                        )
                    else:
                        warped = _T._warp_waypoints(tool_data[tool_idx], delta_b)
                    if float(args.grasp_dz_mm) != 0.0:
                        # 末端低姿态稳态浮高补偿：抓取侧 wp0/wp1 再下压 dz(mm)
                        warped = _T._bias_grasp_waypoints(
                            warped, dy=0.0, dz=float(args.grasp_dz_mm) / 1000.0,
                        )
                    segs = _T._build_tool_segments(
                        wps=warped["waypoints"],
                        safe_z=float(args.safe_z),
                        steps_transit=_ctrl_steps(args.f_transit),
                        steps_descend=_ctrl_steps(args.f_descend),
                        steps_grasp=_ctrl_steps(args.f_grasp),
                        steps_settle=_ctrl_steps(args.f_settle),
                        steps_lift=_ctrl_steps(args.f_lift),
                        steps_place_via=_ctrl_steps(args.f_place_via),
                        steps_to_box=_ctrl_steps(args.f_to_box),
                        steps_release=_ctrl_steps(args.f_release),
                        steps_release_settle=_ctrl_steps(args.f_release_settle),
                        steps_lift_after=_ctrl_steps(args.f_lift_after),
                    )
                    ox, oy = float(drop_off[order_no, 0]), float(drop_off[order_no, 1])
                    if spread_on and (ox != 0.0 or oy != 0.0):
                        segs = _offset_box_segments(segs, ox, oy)
                    # 标签里**必须**带真实工具索引(#N)：顺序随机后"第几把" ≠ "哪把工具"，
                    # 而 guard/steps 都靠标签反查工具索引，靠序号反推会张冠李戴。
                    prefix = (
                        f"收{order_no + 1}/{len(order)}-{TOOL_NAMES[tool_idx]}#{tool_idx}"
                    )
                    for s in segs:
                        s["label"] = f"{prefix}-{s['label']}"

                    tool_start = cum_steps
                    tool_steps = sum(int(s["steps"]) for s in segs)
                    last_release_quat = list(warped["waypoints"][-1]["quat"])
                    place_arm_at[tool_start + tool_steps - 1] = int(tool_idx)
                    all_segments.extend(segs)
                    cum_steps += tool_steps

                # 末把后撤离到箱前安全点 + 入箱检查等待窗口(让最后一把接触验证跑完)
                box_xy_min = np.asarray(
                    box_bottom.get("box_xy_min", box_bottom["xy_min"]),
                    dtype=np.float64,
                )
                box_xy_max = np.asarray(
                    box_bottom.get("box_xy_max", box_bottom["xy_max"]),
                    dtype=np.float64,
                )
                retreat_x = float(0.5 * (box_xy_min[0] + box_xy_max[0]))
                retreat_y = float(box_xy_min[1] - 0.12)
                retreat_pos = [retreat_x, retreat_y, float(args.safe_z)]
                retreat_steps = _ctrl_steps(args.f_retreat)
                all_segments.append(
                    {
                        "steps": retreat_steps,
                        "l_hold": True,
                        "r_target_b": retreat_pos,
                        "r_quat_b": last_release_quat,
                        "gripper_r": "open",
                        "label": "S-撤离箱上方",
                    }
                )
                wait_steps = place_timeout_steps + 2 * steps_scale
                all_segments.append(
                    {
                        "steps": wait_steps,
                        "l_hold": True,
                        "r_target_b": retreat_pos,
                        "r_quat_b": last_release_quat,
                        "gripper_r": "open",
                        "label": "S-入箱检查等待",
                    }
                )
                cum_steps += retreat_steps + wait_steps

                total_steps = cum_steps
                duration_s = total_steps * env_dt
                print(
                    f"\n>>> [尝试{attempt_no}] 场上{len(plan['plant_ids'])}把/收{len(order)}把 | "
                    f"{' '.join(layout_txt)}", flush=True
                )
                yaw_txt = "  ".join(
                    f"{TOOL_NAMES[t]}:{np.degrees(yaw[t]):+5.1f}°" for t in order
                )
                print(f"    [yaw] {yaw_txt}", flush=True)
                print(
                    f"    指令: {prompt}", flush=True
                )
                print(
                    f"    轨迹 {len(all_segments)} 段 / {total_steps} 步(约 {duration_s:.1f}s sim)",
                    flush=True,
                )

                this_smoke = smoke_budget > 0
                if this_smoke:
                    smoke_budget -= 1
                    _dump_frames("pre", ep_display)

                # 预构建完整轨迹并运行
                try:
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = (
                        scripted.build_segmented_trajectory(
                            env, agent_conf, all_segments, g_open_global, g_close_global
                        )
                    )
                except Exception as e:
                    orca_logger.error(f"[轨迹] 构建失败: {e}")
                    stats["fail"]["other"] += 1
                    storage.clear_data()
                    continue

                # 守护设备：常开 S5 抬升空抓早中止；沉降到位再闭爪；--grasp_probe 另打点
                strip_z_by_tool = {int(ti): float(actual[ti][2]) for ti in order}
                settle_cfg = (
                    (float(args.settle_tol_mm) / 1000.0, int(args.settle_max_extra))
                    if args.settle_tol_mm > 0
                    else None
                )
                probe_at, lift_at, settle_at = _build_guard_steps(
                    all_segments,
                    None if args.no_lift_check else strip_z_by_tool,
                    settle_cfg,
                )
                if not args.grasp_probe:
                    probe_at = None
                step_tool = _build_step_tool(all_segments)
                tool_id_holder = {"v": -1}
                if storage is not None and hasattr(storage, "set_tool_id_source"):
                    storage.set_tool_id_source(lambda: tool_id_holder["v"])
                device = _LiftGuardDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                    place_arm_at=place_arm_at,
                    box_bottom=box_bottom,
                    base_body_query=base_body,
                    place_timeout_steps=place_timeout_steps,
                    lift_check_at=lift_at or None,
                    probe_at=probe_at,
                    settle_close_at=settle_at or None,
                    g_open=g_open_global,
                    step_tool=step_tool,
                    tool_id_holder=tool_id_holder,
                )
                manager.set_device(device)
                manager.run_episode()

                if this_smoke:
                    _dump_frames("post", ep_display)
                placed = not device.place_failed
                if not placed:
                    reason = device.place_fail_reason or "放置验证未通过"
                    stats["fail"]["place"] += 1
                    orca_logger.warning(f"[✗] 第{attempt_no}次 全收失败: {reason}，清缓存重采")
                    print(f"  [✗] 未通过放置验证: {reason}，重采", flush=True)
                    storage.clear_data()
                    continue

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success_run += 1
                stats["success"] += 1
                attempts_since_save = 0
                giveup_streak = 0
                print(
                    f">>> [✓] Episode {ep_display}/{n_target} 已保存"
                    f"(数据集累计 {writer.num_episodes} 集 / {writer.num_frames} 帧)",
                    flush=True,
                )
                orca_logger.info(
                    f"[✓] Episode {ep_display} 保存成功 (尝试{attempt_no})"
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断(已保存集数保留，可 --resume 续采)", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = (
            f"采集结束: 本run成功 {stats['success']} 集 / 尝试 {stats['attempt']} "
            f"(place失败 {stats['fail']['place']}, layout {stats['fail']['layout']}, "
            f"giveup {stats['fail']['giveup']}, other {stats['fail']['other']}); "
            f"数据集累计 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        )
        orca_logger.info(summary)
        print(f"\n{'=' * 68}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {args.lerobot_out}", flush=True)
        print("=" * 68, flush=True)
        try:
            scene_manager.show_ui_message(1, "", showtime=0)
            env.render()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("已收到中断请求")
    except Exception as e:
        orca_logger.error(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        orca_logger.info("程序已退出")
        os._exit(0)
