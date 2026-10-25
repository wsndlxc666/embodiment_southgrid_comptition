"""Clear-All(5把全收) · pi0.5-LoRA 策略闭环自测（无落盘）。

用法（先起模型服务，再起本脚本连仿真）：
    # 1) openpi 侧起 websocket serve
    #    cd /home/dgl/lxc/openpi && uv run python scripts/serve_policy.py \
    #        pi05_g1_omnipicker_clear5_lora --exp-name=<exp> --checkpoint-step=<N> --port 8000
    # 2) 本脚本
    #    python g1_omnipicker_policy_clear_eval.py --trials 8 --port 8000

每试：种 5 把工具到"条带标称 ± 小抖动"（与 clear5 采集同分布），指令用采集时同一条
恒定全收句；由策略闭环驱动，直到 5 把全部入箱或预算耗尽。
判定与采集一致：逐把用 _classify_tool_place_contacts 的 ok(箱内面/已入箱工具白名单)
标记"已入箱"，结束再物理兜底(工具 body 在箱底有效区且非仍在爪中)。
一集成功的判据 = 5 把全部标记已入箱。逐把记录完成顺序(越左的越早说明越接近左到右)。

回放节奏（--exec_mode / --action_repeat，改之前务必先读这段）
-----------------------------------------------------------
训练集的 1 帧 = **0.05 s 仿真时间**，依据两条互相独立的证据：
  * 采集器 `clock="sim"`、`--fps 20`，`lerobot_data_storage.collection_data` 用
    `env.data.time`(仿真时间)做门控，每 1/20 s 记一帧；
  * 环境 `env_dt = time_step(0.001) × frame_skip(5) = 0.005 s`。
  ⇒ **1 帧 = 10 个 env 步**；2101 帧/集 = 105 s 仿真时间。
  * 反证：按 20fps 读，训练演示的中位末端速度是 81 mm/s、抓取前下探 36 mm/s，
    对"小心插拔工具"完全合理；按 200fps 读则变成 0.8 m/s、峰值 22 m/s，不可能。

所以正确的回放是：**每个模型动作保持 10 个 env 步，整块 20 步依次下发**
（20 动作 × 0.05 s = 1.0 s 仿真时间，与训练同速），发完再问下一次。

--exec_mode 两种：
  * `chunk`(默认) 如上，整块消费。每次推理覆盖 1.0 s 仿真时间。
  * `first` 旧行为：只下发第 0 步、其余 19 步丢弃，每 `--query_every` 个 env 步重问。
    保留它是为了能和历史结果对照，**不是推荐用法**：单步位移只有 19 mm，而服务端每次
    推理都重采样 flow-matching 噪声(sd 13~30 mm)，信号与噪声同量级，等于原地随机游走。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

_openpi_client_src = "/home/dgl/lxc/openpi/packages/openpi-client/src"
if _openpi_client_src not in sys.path:
    sys.path.insert(0, _openpi_client_src)

base_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
for p in (base_dir, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

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
    capture_frame_with_idx,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import (
    G1OmniPickerLeRobotStorage,
    check_server_state_dim,
)
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

import g1_omnipicker_collection_scripted_tool_lerobot as _T
import g1_omnipicker_collect_clear5_lerobot as C5

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_policy_clear_eval_stream"
_L_INIT = [0.0] * 7
_G_OPEN = -0.8561
_G_CLOSE = 2.0
_VERIFY_POLL = 6
_TOOLBOX_SITE = "Group_Interactive_ToolBox_base_site"
_GOAL_NAMES = list(C5.TOOL_NAMES) + ["盒子"]
# 训练集 fps(见模块 docstring)。用于把 --action_repeat 换算成"每个动作占多少仿真时间"。
_TRAIN_FPS = 20.0
# 第 19 维(goal_dist)的参照点常量。**训练端与推理端必须同一组**：
# 训练时由 scripts/local/build_d19_goal_dataset.py 扫全数据集得出、写进这个 json，
# 推理端只能拿同一组常量重算(v2 用 goals 6 点 = 5 工具 ∪ 盒子落点)。
# v1(tool_refs, 5 点)留着只是让旧代码不报错，新代码不要用。
_DEFAULT_REFS_JSON = os.path.abspath(os.path.join(
    base_dir, "..", "..", "inference", "g1_omnipicker", "tool_refs_d19goal.json"))

# ---- 左臂冻结常量 ----------------------------------------------------------
# 训练集里左臂 7 维(0..6)跨度只有 1.4~18 mm，右臂是 30~93 cm；左爪两维(14,15)
# 跨度**恰好为 0**(q01 == q99 == 0.0480)—— 训练时左臂一动不动、左爪一直张开。
# 推理时模型**原始输出**实测(N=135~600 条 chunk，跨三轮)：
#   左臂位置跨度 15~27 mm、四元数分量跨度 0.045~0.14，左爪跨度恰好 0。
# 也就是说它没到"乱甩"的程度，但确实比训练时(1.4~18 mm)飘得多一点。
# 冻结就是把这多出来的十几毫米和零点几的四元数抖动一并去掉 —— 这是冻结**唯一有依据**
# 的理由：左臂不是这条策略学过要控制的东西，把它的**指令**钉死在训练分布上，
# 就不会有一块模型没学会、却仍在动的自由度。
#
# 【重要·别再说"逐位一致"】冻结改的是**指令**，不是**关节**。2026-09-14 实测：
# 用 state[0:3](= 仿真量到的左臂位姿) 看，冻结轮左臂 x 跨度 43mm / z 51mm、
# 未冻结轮 40mm / 47mm，**几乎没差别**；state 标准差 7.0/0.9/7.8mm(冻结) vs
# 6.3/1.4/7.2mm(未冻结)。而模型原始输出的左臂标准差只有 3mm。
# 即：把 3mm 的指令抖动拿掉，7mm 的**物理抖动**纹丝不动 —— 后者不由指令驱动。
# 而且它与右臂活动强度的相关性只有 +0.03（右臂动得最少的 10% 采样点上，
# 左臂偏移反而更大：9.4mm vs 5.7mm），所以**也不是**右臂反作用力晃的。
# 成因未定。想去掉它只能动物理/控制侧（本脚本 main() 里把两臂 OSC kp 覆写成 220，
# 采集器默认也是 220 —— 但 **④ run_task3_tool_storage_scored.py 用的是 150**，
# 比训练数据软 1.47 倍。上限 300）。动指令没用。
# `_LEFT_POSE_CONST` 保证的是"下发的指令恒等于训练均值"。
#
# 【曾经写在这里、但已被推翻的说法】"冻结能减少左臂摆动给浮动底座的反作用力"。
# 2026-09-14 用同一段代码量了三个转储的**全部** 135 条记录(见 cmp_base 一次性对比)：
#     未冻结(clear5_d19state_0914)  反推底座偏航中位 −0.88°   base 自身移动 [1.77,11.57,0.04] mm
#     冻结  (clear5_freeze_fixed)   反推底座偏航中位 −1.61°   base 自身移动 [2.42,19.80,0.06] mm
#     冻结  (clear5_diag_0914)      反推底座偏航中位 −1.21°   base 自身移动 [2.81,16.66,0.11] mm
# 冻结的两次**不比**未冻结的小，反而略大。而且这个量主要由**右臂**摆动决定，两轮之间
# 右臂轨迹本就不同 → 三个样本、无固定随机种子，**两个方向都证明不了**。
# 所以别再拿"减小底座反作用力"当冻结的理由。（早先报过的 "−0.33° / −3.32mm"
# 是只在某个子窗口上算的，全量复算不成立，已作废。）
#
# 取训练集 state.mean[0:7]，与仿真默认位姿差 <3 mm。
# 重新生成：读 <ckpt>/assets/<repo>/norm_stats.json 的 norm_stats.state.mean[0:7]
# 与 q01[14](=q99[14])。改这个常量前先确认和训练那一档是同一个 checkpoint。
_LEFT_POSE_CONST = np.array([0.132033, 0.976373, 0.302453,
                             -0.500404, -0.498863, 0.500917, 0.499736], dtype=np.float32)
_LEFT_GRIP_CONST = np.float32(0.047967)


def load_tool_refs(path: str | None) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    """读参照点常量。返回 (refs, ref_name, cycle_refs)；refs=None 表示退回 18 维。

    goals(v2, 6 点)优先于 tool_refs(v1, 5 点)：这个 json 两个键都写了，
    只读 tool_refs 会静默退化成 v1 —— 那训练出来的模型收到的是错的输入。

    第三项 cycle_refs：json 里带 "new_dims" 键（= d22 那份 tool_refs_d22.json）时，
    返回里面**有序**的 5 个工具抓取点，storage 据此启用第 20~22 维；旧 json 没这个
    键 → 返回 None → 仍是 19 维，老行为一个字节不变。用 json 自己声明而不是加命令行
    开关：忘了加开关是**静默**错误（state 少 3 维，模型看到从没见过的 prompt），
    而 json 里写没写至少是显式的、能一眼对出来。
    """
    if not path:
        return None, "goal_dist", None
    with open(path, encoding="utf-8") as f:
        tr = json.load(f)
    cycle = None
    if tr.get("new_dims"):
        cycle = np.asarray(tr["tool_refs"], dtype=np.float32).reshape(-1, 3)
    if tr.get("goals"):
        return np.asarray(tr["goals"], dtype=np.float32).reshape(-1, 3), "goal_dist", cycle
    return np.asarray(tr["tool_refs"], dtype=np.float32).reshape(-1, 3), "tool_dist", cycle


def _site_pos(env, name):
    try:
        d = env.query_site_pos_and_quat([name])
        return np.asarray(d[name]["xpos"], dtype=np.float64).reshape(3)
    except Exception:
        return None


def _body_pos(env, name):
    try:
        xpos, _, _ = env.get_body_xpos_xmat_xquat([name])
        return np.asarray(xpos, dtype=np.float64).reshape(3)
    except Exception:
        return None


class ChunkDumper:
    """把每次推理的输入 state / 输出动作块 / 当时的场景几何写成 JSONL。

    为什么需要它：这个脚本原本只在每试结束打一行「入箱 N/5」，600 次推理中间发生了什么
    完全看不见。而动作块 chunk[:, 7:10] 是**右臂末端在 base 系下的目标位置**，与
    tool_refs_d19goal.json 里那 6 个参照点同坐标系同单位 —— 拿它和参照点比 dist 取 argmin，
    就是"这一步想抓哪一件"的直接读数；chunk[:, 16] 是右夹爪指令，能看出何时开合。
    有了这两样才能把仿真里的 600 次决策和训练数据的 5 轮时序逐帧对齐。

    只写文件、不改控制流；任何异常都吞掉，绝不影响评测。
    """

    def __init__(self, path: str, env, l_arm_cfg, r_arm_cfg, echo: str = "brief",
                 exec_mode: str = "chunk", action_repeat: int = 1, env_dt: float = 0.005,
                 refs_path: str | None = None, freeze_left: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")
        self.env = env
        # 和喂给模型的参照点用**同一个**路径，免得"屏上瞄的是 v2、模型吃的是 v1"这种漂移。
        self.refs_path = refs_path
        self.t0 = time.time()
        self.trial = 0
        self.trial_t0 = self.t0
        self.step = 0
        self.query = 0
        self.echo = echo
        # 转储自带执行方式：读的人才知道 chunk 是"整块都下发"还是"只下发第0步"，
        # 否则同一个 chunk 字段在两种模式下含义完全不同。
        self.exec_mode = exec_mode
        self.action_repeat = int(action_repeat)
        self.env_dt = float(env_dt)
        # 同理：转储里的左臂 7 维到底是"模型想让它去哪"还是"实际下发的常量"，
        # 取决于冻结开没开。不记下来，两种转储就无法区分。
        self.freeze_left = bool(freeze_left)
        self.goal_names = list(_GOAL_NAMES)
        self.goals = self._load_goals()
        # conf 里的 ee_site_name / base_body 都是**短名**(如 "body_link1")，
        # 必须先用 env.site() / env.body() 解析成场景里的真实句柄，否则
        # query_site_pos_and_quat / get_body_xpos_xmat_xquat 会直接抛异常返回 None。
        self.ee_site = self._resolve_site(r_arm_cfg["ee_site_name"]) or r_arm_cfg["ee_site_name"]
        self.box = self._resolve_site(_TOOLBOX_SITE) or _T._TOOLBOX_BASE_BODY
        try:
            self.base_body = env.body(agent_conf.base_body)
        except Exception:
            self.base_body = agent_conf.base_body
        self.tool_bodies = list(_T._TOOL_BODY_JOINT_NAMES)

    def _load_goals(self):
        """参照点只用于把动作块读数翻译成'瞄哪一件'，读不到就退化成只打印坐标。"""
        try:
            refs, _, _ = load_tool_refs(self.refs_path)
            return None if refs is None else np.asarray(refs, dtype=np.float64)
        except Exception as e:
            print(f"[dump] 读不到参照点({e})，屏上只打坐标不打'瞄哪件'", flush=True)
            return None

    def _resolve_site(self, name: str):
        try:
            return self.env.site(name)
        except Exception:
            return None

    def begin_trial(self, trial: int) -> None:
        self.trial = trial
        self.trial_t0 = time.time()

    def _pos(self, name):
        p = _site_pos(self.env, name)
        return p if p is not None else _body_pos(self.env, name)

    def write(self, prompt: str, state: np.ndarray, chunk: np.ndarray) -> None:
        try:
            base_w = _body_pos(self.env, self.base_body)
            ee_w = self._pos(self.ee_site)
            box_w = self._pos(self.box)
            tools_w = [self._pos(tb) for tb in self.tool_bodies]
            # 记录离右臂末端最近的那把工具，便于和 runner 的转储格式互相兼容
            nearest = None
            if ee_w is not None:
                cand = [(float(np.linalg.norm(w - ee_w)), w) for w in tools_w if w is not None]
                if cand:
                    nearest = min(cand, key=lambda x: x[0])[1]
            self.fh.write(json.dumps({
                "t": round(time.time() - self.t0, 3),
                "phase_t": round(time.time() - self.trial_t0, 3),
                "trial": self.trial,
                "prompt": prompt,
                "step": self.step,
                "query": self.query,
                "exec_mode": self.exec_mode,
                "action_repeat": self.action_repeat,
                "freeze_left": self.freeze_left,
                "state": [float(x) for x in np.asarray(state).reshape(-1)],
                "chunk": [[float(x) for x in np.asarray(a).reshape(-1)[:18]] for a in chunk],
                "ee_world": None if ee_w is None else [float(x) for x in ee_w],
                "base_world": None if base_w is None else [float(x) for x in base_w],
                "tool_world": None if nearest is None else [float(x) for x in nearest],
                "box_world": None if box_w is None else [float(x) for x in box_w],
                "tools_world": [None if w is None else [float(x) for x in w] for w in tools_w],
                "tool_names": list(_T._TOOL_NAMES),
                "goal_names": list(self.goal_names),
            }, ensure_ascii=False) + "\n")
            self.fh.flush()
            self._echo(chunk, ee_w, base_w, tools_w)
            self.query += 1
        except Exception as e:  # 转储只是诊断，绝不能让它打断评测
            print(f"[dump] 写失败: {e}", flush=True)

    def _echo(self, chunk: np.ndarray, ee_w, base_w, tools_w) -> None:
        """把这次推理的原始输出打到屏幕上（--dump_echo 控制，off 则不打）。"""
        if self.echo == "off":
            return
        a = np.asarray(chunk)
        a0 = a[0, :18]                       # 本块第一步
        rp, rg = a0[7:10], a0[16]            # 右手目标位置 / 右夹爪指令
        last = a[-1]
        ep, eg = last[7:10], last[16]
        travel = float(np.linalg.norm(ep - rp))
        aim = ""
        if self.goals is not None and len(self.goals) > 0:
            d = np.linalg.norm(self.goals - ep, axis=1)
            j = int(np.argmin(d))
            aim = f" | 瞄 {self.goal_names[j]}(距{d[j]*100:.1f}cm)"
        # 手臂此刻实际在哪（= state[7:10]，由 ee_world-base_world 得到，与参照点同系）
        now = ""
        if ee_w is not None and base_w is not None:
            c = np.asarray(ee_w) - np.asarray(base_w)
            now = f"现手({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})→"
        # 说清楚这一块会怎么用：chunk 模式整块依次下发、每步占 action_repeat 个 env 步；
        # first 模式只下发第 0 步。同一个 chunk 在这两种模式下含义完全不同。
        if self.exec_mode == "chunk":
            how = (f"本块{a.shape[0]}步全下发(每步{self.action_repeat}个env步="
                   f"{self.action_repeat * self.env_dt:.2f}s) 起({rp[0]:+.4f},{rp[1]:+.4f},{rp[2]:+.4f})")
        else:
            how = f"只下发第0步({rp[0]:+.4f},{rp[1]:+.4f},{rp[2]:+.4f}) 其余丢"
        print(f"  [q{self.query:3d} t={time.time()-self.t0:6.1f}s tr{self.trial} "
              f"step={self.step:5d}] {now}{how} 爪={rg:+.3f} "
              f"| 块末({ep[0]:+.4f},{ep[1]:+.4f},{ep[2]:+.4f}) "
              f"爪={eg:+.3f} 行程={travel*100:5.1f}cm{aim}", flush=True)
        if self.echo == "full":
            # 原始动作块：20 步 × 前 18 维，逐维打出来
            print(f"  [q{self.query:3d}] 原始块 {a.shape[0]}×{a.shape[1]} "
                  f"(前18维: l_pos0:3 l_quat3:7 r_pos7:10 r_quat10:14 l爪14:16 r爪16:18):", flush=True)
            for k, row in enumerate(a):
                vals = " ".join(f"{v:+.4f}" for v in row[:18])
                print(f"      step{k:02d}  {vals}", flush=True)

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


def _default_joint_values():
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


class PolicyRolloutDevice:
    """从 websocket 策略拿一整块动作 → 按训练节奏依次下发 → 写完再问下一块。

    update() 由 manager.run_controllers() 每个 env 步调用一次：
      * `chunk` 模式(默认)：手上这块 20 步依次下发，每条保持 action_repeat 个 env 步
        (默认 10 步 = 0.05 s = 训练 20fps)，整块走完正好 1.0 s 仿真时间，再问下一次。
      * `first` 模式：只下发第 0 步、其余丢弃，每 query_every 个 env 步重问(旧行为)。
    """
    def __init__(self, l_arm, r_arm, l_grip, r_grip, task_status, env, storage,
                 cameras, camera_map, cam_hw, client, prompt,
                 l_gmin, l_gmax, r_gmin, r_gmax,
                 query_every: int = 10, max_queries: int = 0,
                 exec_mode: str = "chunk", action_repeat: int = 10, dumper=None,
                 freeze_left: bool = True):
        self.dumper = dumper
        self.freeze_left = bool(freeze_left)
        self.l_arm, self.r_arm = l_arm, r_arm
        self.l_grip, self.r_grip = l_grip, r_grip
        self.task_status = task_status
        self.env = env
        self.storage = storage
        self.cameras = cameras
        self.camera_map = camera_map
        self.cam_hw = cam_hw
        self.client = client
        self.prompt = prompt
        self.l_gmin, self.l_gmax = l_gmin, l_gmax
        self.r_gmin, self.r_gmax = r_gmin, r_gmax
        self.query_every = max(1, int(query_every))
        self.max_queries = int(max_queries)   # 0 = 不限，只由外层的步数预算兜底
        self.exec_mode = exec_mode
        self.action_repeat = max(1, int(action_repeat))
        self._n = 0
        self._queries = 0
        self._chunk = np.zeros((0, 18), dtype=np.float32)  # 空块 => 第一次 update 就问
        self._idx = 0
        self._tick = 0
        self.done = False
        self.first_latency = None
        self.last_action = None  # 最后一条动作，结束后保持用

    @property
    def _hold_len(self) -> int:
        """同一条动作保持多少个 env 步再前进。

        first 模式保持 query_every 步(重问闸门)，chunk 模式保持 action_repeat 步。
        """
        return self.query_every if self.exec_mode == "first" else self.action_repeat

    def _query(self) -> bool:
        """问一次策略并把返回的动作块装进缓冲。返回 False 表示该结束了。"""
        t0 = time.time()
        try:
            imgs, _ = capture_frame_with_idx(self.cameras, self.camera_map, self.cam_hw)
            obs = self.storage.obs_callback(self.env)
            state = self.storage.build_state(obs).astype(np.float32)
            payload = {
                "images": {
                    "cam_head": np.asarray(imgs["cam_head"]).astype(np.uint8),
                    "cam_wrist_r": np.asarray(imgs["cam_wrist_r"]).astype(np.uint8),
                },
                "state": state,
                "prompt": self.prompt,
            }
            out = self.client.infer(payload)
            a = np.asarray(out["actions"], dtype=np.float32)  # (H, >=18)
            if a.ndim != 2 or a.shape[1] < 18:
                raise ValueError(f"policy actions 形状异常: {a.shape}")
            if self.dumper is not None:
                # 转储在"每次推理"这一层，不是在每条动作那一层：决策是每次推理一次的，
                # 逐条动作写只会把同一个 chunk 重复 20 遍，白撑大文件。
                self.dumper.step = self._n
                self.dumper.write(self.prompt, state, a)
            if self.first_latency is None:
                self.first_latency = (time.time() - t0) * 1000
            self._queries += 1
        except Exception as e:
            print(f"  [policy] 推理失败: {e}", flush=True)
            self.done = True
            return False
        # first 模式只留第 0 步：整块 20 步会被丢掉 19 步，等于逐帧控制(见模块 docstring)
        self._chunk = a[:, :18] if self.exec_mode == "chunk" else a[:1, :18]
        self._idx = 0
        self._tick = 0
        return True

    def _replan(self) -> bool:
        """到点了就换新块。返回 False 表示已经停下(该结束或推理失败)。"""
        if self.max_queries and self._queries >= self.max_queries:
            self.done = True
            return False
        return self._query()

    def _apply(self, act: np.ndarray) -> None:
        """把一块里的一条动作(18 维)分解成 OSC 目标 + 夹爪指令，写进控制器。

        freeze_left=True 时，左臂 7 维和左爪 2 维被换成常量(见 _LEFT_POSE_CONST)。
        在这里而不是在收到 chunk 时替换，是为了转储里仍保留**模型原始输出** ——
        想知道"模型想让左臂干嘛"看转储，"实际下发什么"看这里，两者不混。
        """
        if self.freeze_left:
            act = np.array(act, dtype=np.float32, copy=True)
            act[0:7] = _LEFT_POSE_CONST
            act[14:16] = _LEFT_GRIP_CONST
        l_pos, l_quat = act[0:3], act[3:7]
        r_pos, r_quat = act[7:10], act[10:14]
        l_g = act[14:16] * (self.l_gmax - self.l_gmin) + self.l_gmin
        r_g = act[16:18] * (self.r_gmax - self.r_gmin) + self.r_gmin
        self.l_arm.update_action_position(l_pos)
        self.l_arm.update_action_axisangle(l_quat)
        self.r_arm.update_action_position(r_pos)
        self.r_arm.update_action_axisangle(r_quat)
        self.l_grip.update_ctrl(np.clip(l_g, self.l_gmin, self.l_gmax))
        self.r_grip.update_ctrl(np.clip(r_g, self.r_gmin, self.r_gmax))
        self.last_action = act

    def update(self) -> None:
        if self.done:
            return
        self._n += 1

        # 手上没块了(第一次，或刚走完最后一条) -> 立刻问，不空转
        if self._idx >= len(self._chunk) and not self._replan():
            return
        # 同一条动作保持 _hold_len 个 env 步，再前进到下一条
        if self._tick >= self._hold_len:
            self._tick = 0
            self._idx += 1
            if self._idx >= len(self._chunk) and not self._replan():
                return
        self._tick += 1
        self._apply(self._chunk[self._idx])

    def force_end(self) -> None:
        self.done = True


def _denorm_range(conf_side):
    n = len(conf_side["actuator_names"])
    rs = conf_side["actuator_ranges"][:n]
    lo = np.array([r[0] for r in rs], dtype=np.float32)
    hi = np.array([r[1] for r in rs], dtype=np.float32)
    return lo, hi


def _tool_in_box(env, tool_body, base_body, placed_set, box_bottom):
    """复用采集器判定：ok(接触全合法) / 箱底有效区物理兜底。

    返回 (是否算入箱, 为什么, 工具在 base 系的位置)。**为什么要返回理由**：
    这个判定有两条会"凭空"判 ok 的路 ——
      (a) `_classify_tool_place_contacts` 在**无接触记录**时不做几何检查，
          只要 COM 落在箱底区域就直接返回 "ok"(工具静止久了会被物理引擎休眠，
          接触查询返回空，正好走进这条)；
      (b) z 容差是相对 z_surface 的 [-0.03, +0.15]，而工具箱底板顶面 z_surface
          和工具躺在桌面上的高度只差几毫米 —— 于是**z 这一维几乎不设防**，
          判定实际上全压在 xy 上。
    所以一旦 xy_min/xy_max 标定得比箱子实际范围宽，桌上的工具就会被误判成"已入箱"。
    打印理由和坐标，误判才能一眼看出来，而不是只看一个数字。
    """
    st, detail, pos_b = _T._classify_tool_place_contacts(
        env, tool_body, base_body, placed_set, box_bottom
    )
    if pos_b is None:
        try:
            pos_b = np.asarray(env.query_position_body_B(tool_body, base_body), dtype=np.float64)
        except Exception:
            pos_b = None
    if st == "ok":
        return True, f"接触判定 ok({detail})", pos_b
    if st == "fail":
        return False, f"接触判定 fail({detail})", pos_b
    # waiting：无接触 → 靠物理位置兜底(在箱底有效区 xy/z 内)
    if pos_b is None:
        return False, "waiting 且拿不到位置", None
    # 直接调采集器**自己**的那份判定，别在这里另写一套：
    # 原来这里硬编码了 z 容差 0.15、xy 余量 0.02，而采集器 `_tool_geom_in_box` 用的是
    # `_BOX_BOTTOM_Z_TOL`(0.10) 和 0.04 —— 于是"什么算在箱里"在项目里有两套定义，
    # 而训练数据是用采集器那套标的。两套不一致时，自测的分和训练的口径就对不上。
    ok = bool(_T._tool_geom_in_box(pos_b, box_bottom))
    # 打印的判定区要和上面**实际用的**余量一致：`_tool_geom_in_box` 的默认 xy_margin=0.04
    # (采集器判定链里就是这么调的)，别拿 _BOX_BOTTOM_XY_MARGIN(0.02，那是给
    # _tool_resting_on_box_bottom 用的)去print，不然打印出来的框比实际判定的小一圈。
    _m = _T._tool_geom_in_box.__kwdefaults__["xy_margin"]
    lo = np.asarray(box_bottom["xy_min"], dtype=np.float64) - _m
    hi = np.asarray(box_bottom["xy_max"], dtype=np.float64) + _m
    dz = float(pos_b[2] - float(box_bottom["z_surface"]))
    return ok, (f"几何兜底({'在箱底区内' if ok else '不在箱底区内'}) "
                f"dz={dz:+.3f}(采集器容差 {_T._BOX_BOTTOM_Z_TOL:+.2f})"
                f"(判定区 x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}])"), pos_b


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 Clear-All(5把全收) 策略闭环自测")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--exec_mode", type=str, default="chunk", choices=["chunk", "first"],
        help="chunk(默认)=整块 20 步依次下发，每步占 --action_repeat 个 env 步，走完再问；"
             "first=旧行为，只下发第 0 步、其余 19 步丢弃。详见模块 docstring 的'回放节奏'。",
    )
    ap.add_argument(
        "--action_repeat", type=int, default=10,
        help="每个模型动作占多少个 env 步。10 = 0.05 s 仿真时间 = 训练 20fps"
             "(env_dt = time_step 0.001 × frame_skip 5 = 0.005 s)。"
             "只在 --exec_mode chunk 下生效；改成别的值就等于让回放速度和训练不一致。",
    )
    ap.add_argument(
        "--sim_budget_s", type=float, default=130.0,
        help="每试给多少秒**仿真时间**。训练一集 = 105 s(5 把全部收完)，130 留 25%% 余量。"
             "注意这是仿真时间不是墙钟时间：本机约 21 env 步/秒，130 s 仿真 ≈ 25 分钟墙钟。",
    )
    ap.add_argument("--max_queries", type=int, default=0,
                    help="策略询问次数硬上限，0=不限(只由 --sim_budget_s 兜底)")
    ap.add_argument("--query_every", type=int, default=10,
                    help="仅 --exec_mode first 下生效：每多少个 env 步重问一次策略")
    ap.add_argument("--task_config", type=str, default="../common/example.yaml")
    ap.add_argument(
        "--tool_refs", type=str, default=_DEFAULT_REFS_JSON,
        help="参照点常量。**必须与训练那一档用同一个文件**，否则模型收到的是它没见过的 state。"
             "d19 模型 → tool_refs_d19goal.json(默认，goals = 5 工具 + 1 盒子, 6 点, state 19 维)；"
             "d22 模型 → tool_refs_d22.json(同一份 json 里多声明了 new_dims，"
             "于是额外启用第 20~22 维 goal_y/cycle_dist/cycle_idx，state 22 维)。"
             "传错了不会静默降级 —— 启动时会拿服务端握手 metadata 里的 state_dim 硬校验并报错。"
             "传空字符串 = 退回 18 维(旧行为，只用于和旧结果对照)。",
    )
    ap.add_argument("--settle_steps", type=int, default=180)
    ap.add_argument("--settle_verify_s", type=float, default=3.0)
    ap.add_argument(
        "--prompt", type=str, default=C5.CLEAR_PROMPTS[0],
        help="固定全收指令(应与训练词族一致；默认=词族第1条)",
    )
    ap.add_argument(
        "--dump_chunks", type=str, default="",
        help="把每次推理的原始输出转储成 JSONL（默认关闭，不影响任何现有行为）。"
             "每条 = 一次推理：state(19 维；--tool_refs 传空时 18 维) + horizon×18 的动作块 "
             "+ 当时的 ee/base/5把工具世界坐标(盒子那个 body 查不到，恒为 null，只用来看)。"
             "动作块第 7:10 维是策略想去右臂末端到哪(base 系)，与 tool_refs_d19goal.json 的参照点"
             "同坐标系，拿它比 dist 取 argmin 就是'这一步在瞄哪一件'；第 16 维是右夹爪指令，能看出何时开合。"
             "转储完用 analyze_chunks.py <文件> 读。",
    )
    ap.add_argument(
        "--dump_echo", type=str, default="brief", choices=["brief", "full", "off"],
        help="转储时同时在屏幕上打每次推理的输出（仅 --dump_chunks 打开时有效）。"
             "brief=一行摘要(手臂现在哪→策略要它去哪/夹爪/瞄准谁)；"
             "full=摘要 + 整个 20×18 动作块逐维打印（600 次会很刷屏）；off=只写文件不打屏。",
    )
    ap.add_argument(
        "--no_freeze_left", action="store_false", dest="freeze_left", default=True,
        help="关掉左臂冻结。默认**冻结**：左臂 7 维 + 左爪 2 维固定成训练集里的常量"
             "(训练时左臂跨度 1.4~18mm、左爪跨度恰好为 0，即一动不动；推理时模型原始输出"
             "实测左臂位置跨度 15~27mm、四元数分量跨度 0.045~0.14，左爪跨度仍恰好 0)。"
             "只为对照才关 —— 冻结的理由是让左臂与训练分布逐位一致，"
             "**不是**减小底座反作用力(那条实测不成立，见 _LEFT_POSE_CONST 上面的注释)。",
    )
    ap.add_argument(
        "--jitter", action="store_true",
        help="打开工具种植随机抖动(复现训练集那种每集布局略不同的分布)。"
             "**默认关闭 = 固定标称布局**：5 把工具都落在 TOOL_REF_POS_B 标称槽位、"
             "yaw=0，每试场景完全一致，便于复现和对比。下面的 --j_*/--big_* 只在开了它之后生效。",
    )
    # 布局抖动（仅在 --jitter 下生效），数值与 clear5 采集默认保持一致
    ap.add_argument("--j_x", type=float, default=0.008)
    ap.add_argument("--j_y", type=float, default=0.018)
    ap.add_argument("--j_yaw_deg", type=float, default=6.0)
    ap.add_argument("--big_frac", type=float, default=0.12)
    ap.add_argument("--big_y", type=float, default=0.040)
    ap.add_argument("--big_yaw_deg", type=float, default=14.0)
    args = ap.parse_args()

    from openpi_client import websocket_client_policy as _wcp
    client = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    # 握手时服务端就把 metadata 发过来了（不额外往返）。里面带它训练时的 state 维度，
    # 等 storage 建好后拿来硬校验 —— 见 check_server_state_dim。
    server_meta = client.get_server_metadata()
    print(f"[策略] 已连接 {args.host}:{args.port}", flush=True)
    print(f"[指令] {args.prompt}", flush=True)

    from yaml import Loader, load as yload
    with open(os.path.join(base_dir, args.task_config), encoding="utf-8") as f:
        scene_config = yload(f, Loader=Loader)
    sm = SceneManager("localhost:50051", config=scene_config)

    dvals = _default_joint_values()
    # state 维度必须和训练那一档一致。pi0.5 把 state **离散成文字**塞进 prompt
    # ("Task: ..., State: v0 v1 ... v18;")，少一维就是每一条训练样本都见过的串短了一个数，
    # 模型看到的是它从没见过的输入 —— 比"某一维被置零"严重得多。
    tool_refs, ref_name, cycle_refs = load_tool_refs(args.tool_refs)
    storage = G1OmniPickerLeRobotStorage(
        dataset_path="/tmp/_eval_scratch", tool_refs=tool_refs, ref_name=ref_name,
        cycle_refs=cycle_refs,
    )
    print(f"[state] {storage.state_dim} 维"
          + (f"（第19维 {ref_name} 参照点 {len(tool_refs)} 个，来自 "
             f"{os.path.basename(args.tool_refs)}）" if tool_refs is not None
             else "（**退回 18 维**，模型训练时是 19 维，此模式下结果不可用于判断策略好坏）")
          + ("（+ 第20~22维 goal_y/cycle_dist/cycle_idx，轮次目标取自该 json 里"
             f"有序的 {len(cycle_refs)} 个工具抓取点）" if cycle_refs is not None else ""),
          flush=True)
    if cycle_refs is None and args.tool_refs:
        print("[state] 注意：这份 json 没声明 new_dims，按 19 维跑。若服务端是 d22 模型，"
              "少 3 维会让模型看到训练时从没出现过的 prompt —— 请改传 tool_refs_d22.json",
              flush=True)
    check_server_state_dim(server_meta, storage.state_dim)
    # 这两行是"这轮跑的是什么"的自述，出问题时先看这里：
    #   左臂到底听不听模型的、工具到底在不在固定槽位。不看这两行很容易把
    #   冻结前后的两次结果混在一起比。
    if args.freeze_left:
        print("[左臂] **冻结**：模型输出里的左臂 7 维 + 左爪 2 维被换成常量，只下发给右臂"
              f"（常量来自训练集 state.mean[0:7]，左爪 {float(_LEFT_GRIP_CONST):.6f}）", flush=True)
    else:
        print("[左臂] 不冻结(--no_freeze_left)：模型原始输出直接下发。只为对照 ——"
              "训练时左臂一动不动，推理时它实测位置跨度 15~27 mm、四元数分量 0.045~0.14", flush=True)
    print(f"[布局] {'抖动(--jitter)：每试重采样工具位置' if args.jitter else '**固定**(默认)：5 把工具都在 TOOL_REF_POS_B 标称槽位、yaw=0，每试完全一致'}"
          f"；共 {args.trials} 试", flush=True)
    manager = DataCollectionManager(
        agent_name="g1_omnipicker", env_name="DataCollection", entry_point=ENTRY_POINT,
        default_joint_values={}, obs_callback=storage.obs_callback, env_index=0, device=None,
        scene_manager=sm, data_storage=None, frame_skip=5, orcagym_addr="localhost:50051",
    )
    env = manager.env
    manager.save_video = False

    env.reset(); time.sleep(0.1)
    if not manager.update_scene():
        print("场景初始化失败"); return
    env.set_default_joint_values(dvals)
    manager.set_disable_actuator_group([agent_conf.positions_group])
    base_body = env.body(agent_conf.base_body)

    ctrl_l = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r, init_r)
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6) * 220.0
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)

    l_gn = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gn = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gn, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gn, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_l, agent_conf.base_body,
                                                    l_gn, init_lg, Controller2F85Reverse.ControllerType.DATA)
    r_grip = create_gripper_2f85_reverse_controller(env, agent_conf.gripper_r, agent_conf.base_body,
                                                    r_gn, init_rg, Controller2F85Reverse.ControllerType.DATA)
    for c in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(c)
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    camera_map = dict(DEFAULT_CAMERA_MAP)
    cameras = {}
    cam_hw = DEFAULT_HW
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        print("camera init fail", e)
    if not cameras:
        print("no cameras; abort"); return

    try:
        box_bottom = _T._calibrate_box_bottom_inner(env, base_body)
        # xy 范围必须打出来：入箱判定的 z 容差是以 z_surface 为基准的 [-0.03, +0.15]，
        # 而 z_surface(工具箱底板顶面)和工具躺在桌面上的高度只差几毫米 —— 于是
        # **z 这一维几乎不设防，判定实际全压在 xy 上**。xy 标定宽一点，桌面上的工具
        # 就会被误判成"已入箱"。不打印这个范围，误判在屏幕上完全看不出来。
        _lo = np.asarray(box_bottom["xy_min"], dtype=np.float64)
        _hi = np.asarray(box_bottom["xy_max"], dtype=np.float64)
        print(f"[放置检测] 箱底标定完成 z_surface={float(box_bottom['z_surface']):.3f}"
              f" | 入箱判定区(base 系) x[{_lo[0]:.3f},{_hi[0]:.3f}] y[{_lo[1]:.3f},{_hi[1]:.3f}]"
              f" 尺寸 {(_hi[0]-_lo[0])*100:.1f}×{(_hi[1]-_lo[1])*100:.1f} cm", flush=True)
        _bx = np.asarray(env.query_position_body_B(_T._TOOLBOX_BASE_BODY, base_body),
                         dtype=np.float64)
        print(f"[放置检测] 工具箱本体在 base 系 ({_bx[0]:.3f},{_bx[1]:.3f},{_bx[2]:.3f})；"
              f"5 把工具的标称槽位在 x[0.508,0.580] y[-0.518,-0.069] ——"
              f"**判定区不该盖住这条带**，盖住了桌上工具就会被误判入箱", flush=True)
    except Exception as e:
        print(f"箱底标定失败: {e}"); return

    env_dt = float(manager.time_step) * int(manager.frame_skip)
    settle_verify_steps = max(1, int(round(args.settle_verify_s / env_dt)))

    # 把回放节奏摊开打印：这是最容易被搞错、又最影响结果的一件事
    if args.exec_mode == "chunk":
        per_q_s = args.action_repeat * env_dt * 20      # 20 = action_horizon(每块步数)
        print(f"[回放] chunk 模式：整块 20 步依次下发，每条动作保持 {args.action_repeat} 个 env 步"
              f"({args.action_repeat * env_dt:.3f}s) => 每次推理覆盖 {per_q_s:.2f}s 仿真时间",
              flush=True)
    else:
        print(f"[回放] first 模式(旧)：只下发第 0 步、其余 19 步丢弃，"
              f"每 {args.query_every} 个 env 步重问 => 每次推理只覆盖 "
              f"{args.query_every * env_dt:.3f}s 仿真时间", flush=True)
    print(f"[回放] env_dt={env_dt:.4f}s；训练集 20fps => 1 帧 = {1/20/env_dt:.0f} 个 env 步；"
          f"每试预算 {args.sim_budget_s:.0f}s 仿真 = {int(args.sim_budget_s/env_dt)} env 步"
          f"（训练一集 105s，5 把全收）", flush=True)

    l_gmin, l_gmax = _denorm_range(agent_conf.gripper_l)
    r_gmin, r_gmax = _denorm_range(agent_conf.gripper_r)
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))

    dumper = None
    if args.dump_chunks:
        dumper = ChunkDumper(args.dump_chunks, env, agent_conf.l_arm, agent_conf.r_arm,
                             echo=args.dump_echo, exec_mode=args.exec_mode,
                             action_repeat=args.action_repeat, env_dt=env_dt,
                             refs_path=args.tool_refs, freeze_left=args.freeze_left)
        print(f"[dump] 逐次推理输出 -> {args.dump_chunks}  (屏幕回显={args.dump_echo})", flush=True)
        print("[dump] 每行格式: [q序号 时刻 tr试 step步] 现手(现在在哪)→这一块怎么下发 "
              "爪=... | 块末(第20步规划终点) 爪=... 行程=... | 瞄 <参照点>(距)", flush=True)

    ok = fail = 0
    n_placed_all = 0
    stat = {}
    t_all = time.time()
    for tr in range(args.trials):
        # 每试开头必须清零轮次计数。不清的话上一试的 5 次开爪会累进来，cycle_idx
        # 直接从 6 起步，模型看到的是训练分布里没有的输入 —— 而且不报任何错。
        storage.reset_cycle()
        if args.jitter:
            xy, yaw = C5._sample_clear_layout(
                rng, j_x=args.j_x, j_y=args.j_y, j_yaw_deg=args.j_yaw_deg,
                big_frac=args.big_frac, big_y=args.big_y, big_yaw_deg=args.big_yaw_deg,
            )
        else:
            # 固定标称布局：抖动全给 0，_sample_clear_layout 直接落回 TOOL_REF_POS_B
            # (与它自己"采不到合法布局"时的兜底是同一条路径)，yaw 也是 0。
            xy, yaw = C5._sample_clear_layout(
                rng, j_x=0.0, j_y=0.0, j_yaw_deg=0.0,
                big_frac=0.0, big_y=0.0, big_yaw_deg=0.0,
            )

        env.reset(); time.sleep(0.05)
        if not manager.update_scene():
            print("场景更新失败，停止"); break
        env.set_default_joint_values(dvals)
        base_body = env.body(agent_conf.base_body)
        C5._plant_tools(env, base_body, xy, yaw)
        C5._settle(env, manager, args.settle_steps)
        env.set_default_joint_values(dvals); env.mj_forward()
        base_body = env.body(agent_conf.base_body)
        actual = _T._query_tool_positions_b(env, base_body)
        good, why = C5._validate_layout(actual)
        if not good:
            print(f"  [skip] 布局校验不过: {why}", flush=True)
            continue

        layout_txt = " | ".join(f"{i}.{C5.TOOL_NAMES[i]}@({xy[i][0]:.2f},{xy[i][1]:.2f})" for i in range(5))
        print(f"\n>>> [trial{tr + 1}] 全收 5 把 | {layout_txt}", flush=True)
        print(f"  prompt: {args.prompt} | layout {why}", flush=True)

        # 控制器初始化：从当前实际末端位起步，避免 OSC 起步乱跳
        manager.set_init_ctrl()
        env.set_ctrl(manager.ctrl)
        for c in manager.controllers:
            c.reset()
        # 与 dataStorage.obs_callback 一致：conf 里的 ee_site_name 是旧名，须先经 env.site()
        # 解析为当前场景真实 site(带 g1_omnipicker_ 前缀)，query 的返回 dict 以解析后句柄为键。
        ee_site_names = [
            env.site(agent_conf.l_arm["ee_site_name"]),
            env.site(agent_conf.r_arm["ee_site_name"]),
        ]
        ee_b = env.query_site_pos_and_quat_B(ee_site_names, [base_body])

        def _ee(name):
            return np.asarray(ee_b[name]["xpos"], dtype=np.float32), \
                np.asarray(ee_b[name]["xquat"][[1, 2, 3, 0]], dtype=np.float32)

        lp, lq = _ee(ee_site_names[0])
        rp, rq = _ee(ee_site_names[1])
        l_arm.update_action_position(lp)
        l_arm.update_action_axisangle(lq)
        r_arm.update_action_position(rp)
        r_arm.update_action_axisangle(rq)
        l_grip.update_ctrl(np.array([_G_OPEN, _G_OPEN], dtype=np.float32))
        r_grip.update_ctrl(np.array([_G_OPEN, _G_OPEN], dtype=np.float32))
        for _ in range(10):
            action = manager.run_controllers()
            env.step(action)
            env.render()

        if dumper is not None:
            dumper.begin_trial(tr + 1)
        dev = PolicyRolloutDevice(
            l_arm, r_arm, l_grip, r_grip, task_status, env, storage,
            cameras, camera_map, cam_hw, client, args.prompt,
            l_gmin, l_gmax, r_gmin, r_gmax,
            query_every=args.query_every, max_queries=args.max_queries,
            exec_mode=args.exec_mode, action_repeat=args.action_repeat,
            dumper=dumper, freeze_left=args.freeze_left,
        )
        manager.set_device(dev)
        base_body = env.body(agent_conf.base_body)

        # 已入箱工具集合：初始 = { }(本集箱内先清空——种植把上集留下的都覆盖走了)
        placed: set[str] = set()
        placed_step: dict[str, int] = {}
        # 预算按**仿真时间**给，不按询问次数：两种 exec_mode 下一次询问覆盖的仿真时间
        # 差 2 倍(0.1s vs 1.0s)，用次数算会让两种模式根本不可比。
        step_budget = int(args.sim_budget_s / env_dt) + 400
        for s in range(step_budget):
            action = manager.run_controllers()  # 内部按 query_every 调 dev.update()
            env.step(action)
            env.render()
            if (s + 1) % _VERIFY_POLL == 0:
                changed = False
                for ti, tb in enumerate(_T._TOOL_BODY_JOINT_NAMES):
                    if tb in placed:
                        continue
                    _ok, _why, _pb = _tool_in_box(env, tb, base_body, placed, box_bottom)
                    if _ok:
                        placed.add(tb)
                        placed_step[tb] = s + 1
                        changed = True
                        _p = "" if _pb is None else f" 工具位置=({_pb[0]:.3f},{_pb[1]:.3f},{_pb[2]:.3f})"
                        print(f"  [已入箱] {C5.TOOL_NAMES[ti]} @step{s + 1} ← {_why}{_p}", flush=True)
                if changed:
                    pass
            if len(placed) == 5:
                print("  [*] 5 把已全部入箱，提前结束", flush=True)
                dev.force_end()
                break
            if dev.done:
                break

        # 策略结束/预算耗尽后：保持其最后动作再沉降验证一段
        if len(placed) < 5:
            for _ in range(settle_verify_steps):
                action = manager.run_controllers()
                env.step(action)
                env.render()
                for ti, tb in enumerate(_T._TOOL_BODY_JOINT_NAMES):
                    if tb in placed:
                        continue
                    _ok, _why, _pb = _tool_in_box(env, tb, base_body, placed, box_bottom)
                    if _ok:
                        placed.add(tb)
                        placed_step[tb] = -1  # 沉降段入箱
                        print(f"  [已入箱] {C5.TOOL_NAMES[ti]} @沉降段 ← {_why}"
                              + ("" if _pb is None else
                                 f" 工具位置=({_pb[0]:.3f},{_pb[1]:.3f},{_pb[2]:.3f})"), flush=True)
                if len(placed) == 5:
                    break

        all5 = len(placed) == 5
        ok += 1 if all5 else 0
        fail += 0 if all5 else 1
        n_placed_all += len(placed)
        # 入箱顺序(按 step)打印；未在任何观察段标记的以沉降段(-1)收尾
        seq = sorted(((placed_step[tb], C5.TOOL_NAMES[i])
                      for i, tb in enumerate(_T._TOOL_BODY_JOINT_NAMES)
                      if tb in placed and placed_step.get(tb, -1) >= 0),
                     key=lambda x: x[0])
        seq_txt = "→".join(n for _, n in seq) if seq else "-"
        print(f"  [{'✓' if all5 else '✗'}] 入箱 {len(placed)}/5 (顺序: {seq_txt}) "
              f"queries={dev._queries}", flush=True)
        stat[all5] = stat.get(all5, 0) + 1

    print(f"\n{'=' * 60}", flush=True)
    print(f"  自测结束: 全收成功 {ok} / 失败 {fail} / {args.trials} 试, "
          f"平均入箱 {n_placed_all / max(1, ok + fail):.1f}/5, {time.time() - t_all:.0f}s", flush=True)
    print("=" * 60, flush=True)
    if dumper is not None:
        dumper.close()
        print(f"[dump] 已写 {args.dump_chunks}", flush=True)
    try:
        env.stop_save_video()
    except Exception:
        pass
    close_cameras(cameras)
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[停止] 中断", flush=True)
    except Exception:
        # 这一层只干一件事：**保证崩溃看得见**。没有它，异常会一路冒到
        # finally 的 os._exit(0) 里 —— 于是退出码是 0、屏幕上看不出出了事，
        # 而"自测结束"那几行又因为 os._exit 不做 flush 而丢在缓冲区里，
        # 结果就是"跑了半小时、日志到一半就没了、还显示成功"。
        traceback.print_exc()
    finally:
        # os._exit 绕过 atexit 和所有缓冲区 flush：重定向到文件时 stdout 是块缓冲，
        # 最后几 KB(恰好是 summary)会整段丢掉。所以退出前自己刷一次。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
