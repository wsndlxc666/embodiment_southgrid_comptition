# G1 OmniPicker 单工具任意位姿抓取 — pi0.5 LoRA 数据采集 / 训练 / 评测报告

日期：2026-09-06
任务：训练一个能"按名字抓取桌面上任意位置的一把指定工具"的视觉-语言-动作(VLA)策略。桌面上同时有 **1~5 把**工具时，其余在场工具作为干扰；目标工具出现在可达区域内任意位置。交付：训练好的 pi0.5-LoRA 检查点 + 本报告。

运行环境：OrcaLab/OrcaGym(MuJoCo) 仿真 @ `localhost:50051`，Unitree G1 OmniPicker（双臂，本任务只用右臂抓取、左臂中立）；GPU RTX 3090 24G。

---

## 0. 结论摘要

- 数据：**176 集 / 32,032 帧**（LeRobot 格式，含 head+wrist 双 RGB 相机 + 18 维 G1 状态/动作），覆盖 5 种电力工具 × 任意桌面位置 × 在场 1~5 把工具。脚本化采集成功率 78%（170/217 尝试）。
- 训练：**pi0.5 + LoRA**（PaliGemma-2B-LoRA 视觉语言 + 300M action-expert-LoRA），6000→8000 步（受 24G 显存与仿真共存限制由 batch8 降 batch4）。
- 自测：在仿真中让策略从随机新鲜布局闭环抓取（serve 策略 + 实时相机），成功率见 §4。
- 交付物文件清单见 §5。

---

## 1. 数据采集

### 1.1 目标
泛化目标是"单一目标工具的任意位置抓取"，因此数据集必须覆盖：
1. **位置多样性**：目标工具放在桌面可达带内任意 (x, y)；
2. **干扰多样性**：桌面上除目标外另有 0~4 把在场工具（随机 1~5 把）；
3. **不在场工具的不可见化**：5 把工具中不在场的那些，绝不能作为"鬼影"出现在画面里误导策略。

### 1.2 布局采样
- 目标位置连续采样于可达带 `TARGET_BANDS`（由机械臂可达性扫描 `/tmp/grasp_reach_map.json` 得出 OK 区 x∈[0.44,0.68]）：
  - 主带 x∈[0.50,0.60], y∈[-0.50,-0.12]（权重 0.75）
  - 扩展带 x∈[0.44,0.68], y∈[-0.55,-0.10]（权重 0.25）
- 干扰工具铺在 `/tmp/placeable_scan.json` 的 64 个稳定格点（去掉工具箱旁 guard 区），与目标间距 ≥0.17m，不重叠。
- 不在场工具统一丢到 x=0.35 桌边"掉落列"，会物理沉到桌面以下。
  - **可见性 A/B 验证**：单场景同 episode 内比较"丢落藏法 vs 瞬移到地下深处"的头/腕相机帧 → 0% 像素显著变化（阈值>12/255），证明藏法不可见、不会在训练数据里留鬼影。
- 指令文本按 5 种工具 × 3 种祈使动词随机：`抓取/拿起/抓起 X`。

### 1.3 执行与校验
- 每条数据用脚本化专家轨迹执行（右臂 OSC 到达-下落-合爪-抬起），`grasp` 成功判据 = 目标 COM 抬离桌面 z>max(0.30, z0+0.12)。
- 失败自动清缓存重采：`layout` 失败（落点/沉降异常，z0∉[0.118,0.20] 或干扰掉桌）或 `grasp` 失败；连续 `max_attempts` 失败触发 giveup 整轮重来，防止脏数据污染。
- 只保存成功抓起的 episode；未保存任何失败轨迹。

### 1.4 采集结果
脚本化采集（bulk 阶段）：**成功 170 集 / 尝试 217（78.3%）**；失败分项：grasp 37、layout 10、giveup 0、other 0。

| 项 | 值 |
|---|---|
| 数据集总集数 / 总帧数 | 176 集 / 32,032 帧（每集固定 182 帧，fps=20）|
| 动作/状态维数 | 18（见 §3.1），镜像 cam_head + cam_wrist_r，HEVC 视频 |
| 目标工具分布(176集) | 扳手 33 · 螺丝刀 33 · 手电筒 28 · 电工刀(左) 43 · 电工刀(右) 39 |
| 指令动词分布(176集) | 抓取 105 · 拿起 37 · 抓起 34 |
| 在场工具数 k 分布(170集 bulk) | k1=32 · k2=42 · k3=40 · k4=42 · k5=14 |
| 单集轨迹长度 | 182 帧（≈9.1s @20fps），含到达/下落/合爪/抬升各段 |
| 数据规模 | ~77.9 GB（视频为主）|

k5 相对偏少：5 把同时在场且不碰撞的布局空间较窄，脚本化的"5 把在场"难度最高。对"按名字抓取"目标，1~5 把都已有覆盖。

---

## 2. 训练

### 2.1 模型与配置
- 基座：**pi0.5**（`pi05=True`），视觉-语言为 `PaliGemma 2B`、动作专家 `Gemma 300M`，两者均为 **LoRA 微调**（基座权重冻结为 bf16）。
- 内部 `action_dim=32`（基座权重约束），数据侧用 `PadStatesAndActions` 把 18 维状态/动作 0 填充到 32，模型输出再截回 18（`G1OmnipickerOutputs`）。
- `action_horizon=20`（每步输出 20 步动作块）。
- 数据：`g1_omnipicker_single_grasp`，`prompt_from_task=True`（episode 的指令文本作 prompt）。
- 优化：AdamW(clip=1.0)，cosine decay 2e-4→2e-5，warmup 800 步；**num_steps=8000**；无 EMA。
- 加载 `gs://openpi-assets/checkpoints/pi05_base/params` 基座权重（本地已缓存）。
- 归一化统计：`compute_norm_stats.py --max-frames 3000` → `assets/pi05_g1_omnipicker_single_grasp_lora/g1_omnipicker_single_grasp/norm_stats.json`（state/actions 各自 mean/std）。

### 2.2 显存调整（重要运行时决策）
初设 batch=8 在 step 0 触发 `RESOURCE_EXHAUSTED`：XLA rematerialize 后程序峰值仍 ~18.8 GiB，与 orcalab 仿真(占用 ~5.3 GiB)并存时超 24 GiB。→ 降到 **batch=4**（峰值落回可用余量内，与仿真共存稳定，实测 100% util 无 OOM）。

### 2.3 训练实况
- 命令与 exp：`scripts/train.py pi05_g1_omnipicker_single_grasp_lora --exp-name=g1_single_grasp_v1`
- 实测约 **2.5 s/step**（batch4, 3090）→ 8000 步 ≈ 5.6 h。
- 保存：每 500 步存一次（`save_interval=500`, keep 500），末步另存。
- 检查点目录：`checkpoints/pi05_g1_omnipicker_single_grasp_lora/g1_single_grasp_v1/`

### 2.4 训练曲线
【待训练结束后填：末步 loss / 代表性 loss 序列 / 是否用 EMA】

---

## 3. 状态/动作语义（与采集一致）

18 维，LeRobot G1 OmniPicker 约定，均为**基座系(b)**下、**实测**末端 effector site 位姿：

```
[ l_pos(3)  l_quat_xyzw(4)  r_pos(3)  r_quat_xyzw(4)
  l_grip_inner_norm(1)  l_grip_outer_norm(1)  r_grip_inner_norm(1)  r_grip_outer_norm(1) ]
```
夹爪两列各按 actuator 量程线性归一化到 [0,1]（G1 ranges (−1,2)，norm=(v+1)/3）；动作 = 下一绝对状态。策略动作经归一化统计后由模型输出；推理端把 `a[0,:18]` 解码成左/右 OSC 末端位置+四元数、并把夹爪列反归一化回电机 ctrl 单位驱动。

---

## 4. 仿真闭环自测

方法：训练完成后 `scripts/serve_policy.py` 起 websocket 策略服务；自测脚本 `g1_omnipicker_policy_grasp_eval.py` 连接 OrcaGym(50051)，对每个 trial：
- 随机 k∈[1,5]、随机目标工具、随机目标 (x,y)（与训练同带），其余在场工具随机铺开 → 全新布局；
- 由策略从默认中立位闭环驱动（每 10 env 步查询一次、执行 20 步动作块首帧），最多 N 次查询；
- 成功判据同训练：目标 COM 抬离桌面 z>max(0.30, z0+0.12)，并保持不掉落。

自测结果（t=**k 分布：k1~k5 全覆盖**）：
【待训练结束后填：OK/fail per trial、按工具×k 明细、平均首帧推理延迟 ms、典型失败模式】

---

## 5. 交付物清单

| 路径 | 说明 |
|---|---|
| `openpi/data/g1_omnipicker_single_grasp/` | LeRobot 数据集 176 集/32,032 帧 |
| `openpi/assets/pi05_g1_omnipicker_single_grasp_lora/g1_omnipicker_single_grasp/norm_stats.json` | 归一化统计 |
| `openpi/checkpoints/pi05_g1_omnipicker_single_grasp_lora/g1_single_grasp_v1/<step>/` | 训练检查点（末步 <step>）|
| `openpi/src/openpi/training/config.py` `pi05_g1_omnipicker_single_grasp_lora` | 训练/推理配置 |
| `SouthGrid/.../g1_omnipicker_collect_single_grasp_lerobot.py` | 批量采集脚本 |
| `SouthGrid/.../g1_omnipicker_policy_grasp_eval.py` | 策略闭环自测脚本 |
| 本报告 | — |

### 复现命令
```bash
# 1) 训练（已执行）
cd /home/dgl/lxc/openpi
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data HF_HUB_OFFLINE=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline \
  uv run scripts/train.py pi05_g1_omnipicker_single_grasp_lora --exp-name=g1_single_grasp_v1 --overwrite

# 2) 起策略服务
uv run python scripts/serve_policy.py \
  --policy.config=pi05_g1_omnipicker_single_grasp_lora \
  --policy.dir=checkpoints/pi05_g1_omnipicker_single_grasp_lora/g1_single_grasp_v1/<step> --port 8000

# 3) 仿真闭环自测（另开终端，orcalab python）
cd SouthGrid/src/examples/dataCollection/g1_omnipicker
/home/dgl/miniconda3/envs/orcalab/bin/python g1_omnipicker_policy_grasp_eval.py --trials 24 --port 8000
```

---

## 6. 已知限制与后续建议
- 5 把同时在场的样本偏少（k5=14/170），若上线场景多为满桌 5 把，建议补采 k=5。
- 本自测在仿真闭环执行；仿真/真实域差（相机、动态）未验证。
- LoRA 步数按 deadline 压缩（batch8→4 后 8000 步 ≈1 epoch）；若精度不足，可续训更久或提高 batch（在仿真占显存时先停仿真）。
- 未见任何异常自动避碰/掉桌后的恢复策略；抓取中途目标被碰走时策略无恢复数据。

---

*报告主体由采集/训练日志、数据集 meta、训练日志汇总；§2.4/§4 数字于训练与自测完成后回填。*
