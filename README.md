# 南方电网具身智能大赛 · 参赛材料汇总

本仓库汇总本队（`csgComp_Team_08`）参加南方电网具身智能大赛的三部分工作：任务一（G1 移动巡检）、任务三（G1 OmniPicker 工具收纳）、以及支撑任务三的数据采集 / π0.5 模型训练 / 闭环推理代码。大赛官方说明文档与 SouthGrid 实现指南也一并收录在 `docs/`。

> 这是一个**代码与文档快照**。模型 checkpoint、训练数据集、虚拟环境和 wandb 记录体积达数百 GB，未纳入本仓库，清单见下文「未纳入的大文件」。

## 目录结构

```
southgrid-competition/
├── README.md                  # 本文件
├── docs/                      # 大赛官方文档
│   ├── 南方电网具身智能大赛详细说明文档_v8.5.2.pdf
│   └── SouthGrid端到端实现指南_v3.pdf
├── southgrid/                 # SouthGrid SDK 工作副本（仿真 / 采集 / 推理 / 任务三脚本）
├── openpi/                    # openpi（π0.5）训练与推理代码副本，含本队改动
├── task1_locomotion/          # 任务一：G1 + HEFT 行走与检查点巡检
└── submission/                # 任务三最终提交包（技术文档 + code/）
```

## 各部分说明

### `southgrid/`

主办方 SouthGrid SDK 的工作副本，保留原始目录结构：

- `src/model/`：`G1_omnipicker`、`G1_Pick` 的 MuJoCo XML 与 mesh 资产
- `src/examples/dataCollection/g1_omnipicker/`：数据采集器（含 clear5 四项随机化采集、批量看门狗、回放）
- `src/examples/inference/g1_omnipicker/`：在线推理、任务三评分驱动、相位/IK/几何分析探针
- `docs/`：官方采集与推理说明
- `_attic/`：早期脚本存档（保留备查，非主流程）

主要任务三入口：

| 用途 | 文件 |
|---|---|
| 数据采集（四项随机化） | `src/examples/dataCollection/g1_omnipicker/g1_omnipicker_collect_clear5_lerobot.py` |
| 批量采集看门狗 | `src/examples/dataCollection/g1_omnipicker/run_clear5rand_bulk.sh` |
| 闭环推理 + 评分 | `src/examples/inference/g1_omnipicker/run_task3_tool_storage_scored.py` |
| 策略服务启动 | `src/examples/inference/g1_omnipicker/serve_clear5rand.sh` |

环境安装、资产订阅与运行方式见 `southgrid/README.md` 及其 `docs/`。

### `openpi/`

基于 [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi)（Apache-2.0）的 π0.5 LoRA 微调工作副本，针对 G1 OmniPicker 工具收纳改造：

- `src/openpi/policies/g1_omnipicker_policy.py`：机器人输入/输出映射
- `src/openpi/training/config.py`：新增 `pi05_g1_omnipicker_*` 系列训练配置
- `src/openpi/models/pi0.py`、`pi0_config.py`：自定义 state 维度、按维损失权重等改动
- `scripts/train.py`、`scripts/serve_policy.py`：训练与策略服务入口

训练数据与权重不在仓库内。改动摘录与复现说明另见 `submission/code/training/openpi_patch/`。

### `task1_locomotion/`

任务一（移动巡检 A/B/C 检查点）运行包：HEFT G1 行走策略（ONNX）、动作片段、评分客户端接入与导航脚本。

- 入口：`run_task1_heft_navigation.py`
- 配置：`nav_task1.yaml`
- 运行：`run_exp.sh`
- 说明：`README.md`、`docs/`、`ORCA_DEVELOPMENT_GUIDE.md`

上游仓库：<https://codehub.devcloud.cn-east-3.huaweicloud.com/bbb453d36204472ca668f7c8ca786e59/SouthGrid_Unitree_g1_locomotion.git>（本副本不含其 Git 历史）

### `submission/`

任务三最终提交内容：`技术实现文档_任务三.md` 与 `code/`（采集 / 训练 / 推理三段）。`README_上传说明.md` 记录了提交时的状态与注意事项。

## 未纳入的大文件

| 内容 | 原路径 | 体积 | 说明 |
|---|---|---|---|
| 训练产出的 checkpoint | `openpi/checkpoints/` | ≈163 GB | 各实验的 LoRA 权重（JAX） |
| 训练数据集 | `openpi/data/` | ≈15 GB | LeRobot 格式，含 clear5 / clear5_rand 等 |
| Python 虚拟环境 | `openpi/.venv/` | ≈7.9 GB | 按 `openpi/pyproject.toml` 重建 |
| wandb 记录 | `openpi/wandb/` | ≈18 MB | 训练曲线 |
| 会话 / 运行记录 | `**/.runlogs/` | ≈12 MB | 已从本仓库移除 |

需要数据集或权重时，可按 `submission/技术实现文档_任务三.md` 的命令重新采集 / 训练，或向本队索取。

## 快速开始

```bash
# 1) SouthGrid 环境
cd southgrid
conda env create -f environment-unitree.yml
conda activate orcalab_lerobot
bash scripts/install_runtime.sh

# 2) openpi 环境（训练 / 策略服务）
cd ../openpi
uv sync

# 3) 任务一
cd ../task1_locomotion
./run_exp.sh
```

具体任务布局、资产订阅与评测入口以 `docs/` 中两份官方文档和各自 README 为准。

## 安全说明

- 仓库内的队伍 token 已全部替换为 `<你的队伍token>` 占位符，无硬编码凭据。
- 会话记录 `.runlogs/` 与训练数据未上传。

## 来源与许可

| 部分 | 来源 | 许可 |
|---|---|---|
| `openpi/` | Physical Intelligence openpi | Apache-2.0 |
| `southgrid/` | openverse-orca SouthGrid | MIT |
| `task1_locomotion/` | SouthGrid_Unitree_g1_locomotion | MIT |
| `docs/` | 大赛主办方 | 版权归主办方，仅供参赛使用 |
