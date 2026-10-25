# G1 OmniPicker 按名"抓起指定工具 + 放入工具箱" — pi0.5 LoRA 两阶段训练报告

日期：2026-09-06
任务（最终确认范围）：桌面上同时有 **1~5 把** 电力工具时，按语言指令（含目标工具名）
把**指定工具从桌面任意可达位置抓起并搬运放入工具箱**；其余在场工具作为干扰。
交付：能闭环执行该 pick+place 的 pi0.5-LoRA 检查点 + 本报告 + 数据/脚本清单。

运行环境：OrcaLab/OrcaGym(MuJoCo) 仿真 @ `localhost:50051`，Unitree G1 OmniPicker
（右臂执行、左臂中立）；GPU RTX 3090 24G（与仿真共存，训练 batch=4）。
工具箱在基座系固定于 ~(0.587, 0.19)。

---

## 0. 结论摘要

- **范围修正（重要）**：初版任务理解为"单工具任意位姿抓取"，完成并训练了抓取模型后，
  用户确认真实任务需 **完整抓起+放箱**。因此本报告记录两阶段做法：
  - **阶段 A（已完成）** 任意位姿**抓取**模型：数据 176 集 → 训完 8000 步；
  - **阶段 B（本次交付核心）** 在其上继续微调，把后半段续成 **搬运→入箱→松开**。
- 阶段 B 数据：**【待回填】** 集（抓起+放回工具箱，含 1~5 把在场干扰、失败自动重采剔除）。
- 阶段 B 训练：热启动自阶段 A 末检查点（7999）的 LoRA 微调，步数【待回填】。
- 自测：仿真新鲜随机布局下按名闭环 pick+place，入箱成功判据=工具 body 与箱底内面
  有效接触。结果见 §4。
- 交付物清单见 §5。

---

## 1. 数据采集（两阶段）

### 1.1 阶段 A —— 任意位姿抓取（热启动底座）
目标位置连续采样于可达带 `TARGET_BANDS`（x∈[0.44,0.68] 为主，见旧报告），覆盖
5 工具 × 任意桌面位置 × 在场 1~5 把；不在场工具藏入桌下掉落区（可见性 A/B 已验证不可见）。
指令为纯抓取动词 `抓取/拿起/抓起 X`。**成功 170 集/尝试 217（78.3%）**，
数据集 `g1_omnipicker_single_grasp`：**176 集 / 32,032 帧**（每集 182 帧）。
该模型在任意位置抓取上已足够泛化，故用作阶段 B 的热启动；其"抓取后停住保持"的
末端行为将在阶段 B 被"搬运入箱"覆盖。

### 1.2 阶段 B —— 完整 pick+place（本次新采集）
- 工具箱基座系固定 → 5 件工具的路点 YAML 均为 **6 路点**且 `wp2..wp5`（放箱段：
  via1/via2/箱上/松开）**跨工具完全相同**；只有 `wp0/wp1`（接近/抓取）随工具不同。
  → 任意落点只需把 `wp0/wp1` 平移 delta 到实际位置，放箱段保持模板固定即可复用同一套
  入箱几何（脚本 `_T._warp_waypoints` 正是只平移 wp0/wp1）。
- 轨迹分段复用工具整理采集的 `_T._build_tool_segments`（高位过渡→下降→对准抓取→闭爪
  沉降→抬升→放箱经由×2→移到箱上→逼近松开位→沉降松开→松开后抬升），末尾追加
  **撤离段 + 入箱检查等待段**（等待窗 ≥ 放置验证超时，保证监视器在轨迹结束前得出判定）。
- **成败检测（每尝试逐次判定，非抽查）**：`G1ScriptedTrajectoryDevice` 入箱状态监视器——
  目标工具路点结束后，等 EE xy 撤出整箱矩形，再逐控制步对目标工具 body 做接触白名单分类：
  - 合法接触 = 箱底内面 / 已入箱工具 → 放置有效；
  - 非法接触（机器人/桌面/货架/掉落的其它物）或超时(2.5 s)无有效接触 → `place_failed=True`
    → 清缓存、统计、换布局重采。
- **为何剔除失败轨迹（不入训练）**：本模型是纯离线行为克隆，无"负标签"机制——轨迹一旦进
  数据集，每一帧都会被当作正确示范去拟合。失败案例的上下文（目标工具/布局/prompt）常与
  成功案例同构，喂进去等于同一观察给两条矛盾的动作续集，会拖糊已经学好的行为；且失败尾段
  （空爪在空中、工具掉桌被推）会污染归一化统计。失败只留日志做诊断（z 前后 + 原因 + 布局），
  用于统计成功率与定位难点，不参与训练。LeRobot/pi0 脚本化专家数据惯例亦同。
- **指令措辞刻意与阶段 A 分开**（`把X放回工具箱` / `抓起X放入工具箱` / `将X放进箱子里`）：
  阶段 A 的同文本已学成"抓后停住"，阶段 B 用不同指令 token 让模型在语言层分支，避免
  两套续集在同一 prompt 上冲突。

采集结果（阶段 B，脚本 `g1_omnipicker_collect_single_place_lerobot.py`，输出
`g1_omnipicker_single_place`）：
**【待回填：总集数/帧数、成功率(成功/尝试)、place 失败分项、每工具分布、k 分布、单集平均耗时】**
冒烟 4 固定布局 + 随机回退：4/6 尝试成功（含 k=1 与 k=5 场景），失败两例分别出现在
近箱位(0.58,−0.16)与刀类工具，均被正确检出并重采。

---

## 2. 训练

### 2.1 模型与配置（两阶段同构）
pi0.5（`pi05=True`）：PaliGemma-2B-LoRA 视觉语言 + Gemma-300M-LoRA 动作专家，
基座冻结 bf16。内部 `action_dim=32`（0 填充/裁剪到 18），`action_horizon=20`。
数据 `prompt_from_task=True`。AdamW(clip=1.0)，cosine 2e-4→2e-5。

| 项 | 阶段 A | 阶段 B |
|---|---|---|
| 配置名 | `pi05_g1_omnipicker_single_grasp_lora` | `pi05_g1_omnipicker_single_place_lora` |
| 数据 | g1_omnipicker_single_grasp(176 集) | g1_omnipicker_single_place(【回填】集) |
| 起点 | 基座 pi05_base | 阶段 A 7999 检查点（热启动）|
| batch / 步数 | 4 / 8,000 | 4 / 【回填】 |
| warmup/peak/decay | 800/2e-4/8e-4 步 | 【回填】 |
| norm_stats | 已生成 | 【回填：生成路径】 |

> 显存说明：batch=8 在 3090 24G 与仿真共存时 step0 OOM（需 ~25GB）→ 降到 batch=4
> 稳定（~18GB 训练 + ~6GB 仿真 ≈ 24GB 上限），实测 ~2.5 s/it。

### 2.2 训练曲线
【待训练结束后回填：末步 loss、代表 loss 序列、是否 EMA】

---

## 3. 状态/动作语义（与两阶段采集一致）
18 维，LeRobot G1 OmniPicker 约定，基座系(b)下实测末端 effector：
`[l_pos(3) l_quat_xyzw(4) r_pos(3) r_quat_xyzw(4) l_grip_inner l_grip_outer r_grip_inner r_grip_outer]`
夹爪按 actuator 量程归一化 [0,1]；动作 = 下一绝对状态。推理端解码 a[0,:18] → 左/右 OSC
末端位置+四元数，夹爪反归一化驱动。

---

## 4. 仿真闭环自测（阶段 B）
方法：训练后 `scripts/serve_policy.py` 起策略服务；`g1_omnipicker_policy_place_eval.py`
连接 OrcaGym(50051)。每个 trial：
- 全新随机布局（k∈[1,5]、随机目标工具、随机目标(x,y)同训练带、干扰铺开、不在场工具隐藏）；
- 指令从阶段 B 措辞族采样（含目标名）；
- 策略闭环驱动（每 10 env 步查询一次、执行 20 步动作块首帧）；
- 成功判据同采集：目标工具 body 与箱底内面有效接触且无非法接触（必要时用渲染帧人工复核）。

自测结果：【待训练后回填：OK/fail 明细（按工具×k）、平均首帧推理延迟 ms、典型失败模式】

---

## 5. 交付物清单

| 路径 | 说明 |
|---|---|
| `openpi/data/g1_omnipicker_single_place/` | 阶段 B 数据集【回填规模】 |
| `openpi/assets/pi05_g1_omnipicker_single_place_lora/g1_omnipicker_single_place/norm_stats.json` | 归一化统计【回填】 |
| `openpi/checkpoints/pi05_g1_omnipicker_single_place_lora/<exp>/<step>/` | 阶段 B 检查点【回填末步】 |
| `openpi/checkpoints/pi05_g1_omnipicker_single_grasp_lora/g1_single_grasp_v1/7999/` | 阶段 A 检查点（热启动源） |
| `openpi/src/openpi/training/config.py` 两个配置 | 训练/推理配置 |
| `SouthGrid/.../g1_omnipicker_collect_single_place_lerobot.py` | 阶段 B 采集脚本（含逐次成败检测） |
| `SouthGrid/.../g1_omnipicker_policy_place_eval.py` | 阶段 B 闭环自测脚本 |
| `SouthGrid/.../REPORT_single_grasp_2026-09-06.md` | 阶段 A 专报（数据/训练明细） |
| 本报告 | — |

### 复现命令
```bash
# 阶段 A（已完成）训练
cd /home/dgl/lxc/openpi
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
  uv run scripts/train.py pi05_g1_omnipicker_single_grasp_lora --exp-name=g1_single_grasp_v1

# 阶段 B：采集（已执行）
cd SouthGrid/src/examples/dataCollection/g1_omnipicker
/home/dgl/miniconda3/envs/orcalab/bin/python -u g1_omnipicker_collect_single_place_lerobot.py \
  --num_episodes 60   # 逐次成败检测，失败自动重采，只存验证通过的 episode

# 阶段 B：归一化统计
cd /home/dgl/lxc/openpi
HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data uv run python scripts/compute_norm_stats.py \
  --config-name=pi05_g1_omnipicker_single_place_lora --max-frames 3000

# 阶段 B：训练（热启动自阶段 A 7999）
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
  uv run scripts/train.py pi05_g1_omnipicker_single_place_lora --exp-name=g1_single_place_v1

# 阶段 B：自测
uv run python scripts/serve_policy.py \
  --policy.config=pi05_g1_omnipicker_single_place_lora \
  --policy.dir=checkpoints/pi05_g1_omnipicker_single_place_lora/g1_single_place_v1/<step> --port 8000
# （另开终端）
/home/dgl/miniconda3/envs/orcalab/bin/python g1_omnipicker_policy_place_eval.py --trials 24 --port 8000
```

---

## 6. 已知限制与后续
- 【阶段 B 数据量 / 训练步数按截止时间压缩，若闭环不足可续采/续训】
- k=5 满桌场景可能偏少（同阶段 A），上线若多满桌可补 k=5。
- 自测为仿真闭环；仿真→真实域差（相机、动态、箱子几何）未验证。
- 失败轨迹按上文原因剔除；若需"负例"型学习，需引入判别式/分类头或语言负标签框架，超出当前 LoRA-BC 范围。

*本报告数字随阶段 B 采集/训练/自测完成逐步回填。*
