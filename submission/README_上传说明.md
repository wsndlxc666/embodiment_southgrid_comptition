# 任务三提交包 · 上传说明

队伍：`csgComp_Team_08`
截止：2026-09-24 23:59:59（本地时间 2026-09-24 17:15 生成）

---

## 一、上传位置

按大赛须知第 6 条，在**团队信息 → 添加附件**中上传：

| 上传项 | 对应本目录的文件 |
|---|---|
| **技术实现文档** | `技术实现文档_任务三.md` |
| **代码** | `code/` 整个目录（打包成 zip 上传） |

建议把 `code/` 打成 zip：

```bash
cd /home/dgl/lxc/competition_submission
zip -r 代码_任务三.zip code/
```

---

## 二、上传前必须确认的三件事

### ① 分数记录：最后一次提交是不是「脚本」跑的？

大赛须知第 5 条：**任务二和任务三需要经过数据采集、模型训练的方式提交最终的成绩，
非模型训练的得分可能导致扣分甚至零分。**

本队本地的评分记录（`~/.orca_comp_scorer/summary_att_*.json`）显示，
任务三最后两次 attempt 是 **38.89 / 38.93 分（6/6 全过）**，
但它们由 `run_task3_tool_storage_scripted_scored.py` 产生 ——
该脚本**完全不连策略服务**，是纯脚本航点轨迹，**不属于模型能力**。

| 时间 (UTC) | attempt | 方式 | 得分 /40 |
|---|---|---|---|
| 2026-09-11 05:45 | `att_f0f4ed72daae` | **模型** | **18.10（3/6）** |
| 2026-09-12 14:01 | `att_e2278fc7a4ed` | 脚本 | 38.89 |
| 2026-09-12 14:05 | `att_b2b116b8bb3c` | **脚本（最后一次）** | 38.93 |

须知第 2 条：每任务可提交 5 次、系统保留最后 3 次、**成绩以最后一次为准**。
若平台上最后一条记录就是这两次脚本成绩之一，**存在按第 5 条被扣分或清零的风险**。

**行动建议**：用模型方式再提交一次，把最后一条记录覆盖成模型成绩。
模型服务与评分驱动的启动命令见技术文档 §5.1 / §5.2。

> ⚠ **这需要 OrcaLab 仿真在运行**（`localhost:50051`）。
> 撰写本文档时仿真未启动，因此**无法代为完成**。

### ② 本轮的随机化模型（12000 步）**还没有闭环实测得分**

`pi05_g1_omnipicker_clear5_rand_lora` 已训练完成、loss 收敛良好，
但闭环评测尚未进行。技术文档 §8.2 已如实写明这一点，
**没有把它包装成已验证的结果**。

### ③ 视频轨迹

须知第 3 条要求：上传的视频中要有机器人的轨迹，**不要遮挡机器人操作画面**。
本地录屏在 `~/.orca_comp_scorer/recordings/`（形如
`att_<id>_task3_tool_storage.mp4`），确认后一并通过评分客户端上传。

---

## 三、目录结构

```
competition_submission/
├── README_上传说明.md            ← 本文件
├── 技术实现文档_任务三.md         ← 上传项 ①
└── code/                         ← 上传项 ②（打包成 zip）
    ├── collection/               数据采集
    │   ├── g1_omnipicker_collect_clear5_lerobot.py          主采集器（四项随机化）
    │   ├── g1_omnipicker_collection_scripted_tool_lerobot.py 被依赖的工具/几何库
    │   ├── g1_tool.json                                      任务三场景布局
    │   └── run_clear5rand_bulk.sh                            批量采集看门狗
    ├── training/                 模型训练
    │   ├── train_clear5rand.sh              起训脚本（四道闸）
    │   ├── norm_stats_direct.py             归一化统计量（替代官方脚本）
    │   └── openpi_patch/                    openpi 侧改动（见该目录 README.md）
    │       ├── README.md
    │       ├── config_新增条目.py
    │       ├── g1_omnipicker_policy.py
    │       ├── pi0_config.py
    │       └── pi0.py
    └── inference/                推理与评分
        ├── run_task3_tool_storage_scored.py   评分驱动（正式 entry）
        ├── eval_g1_omnipicker_tool_lerobot.py 被依赖的控制器/相机/策略管线
        └── serve_clear5rand.sh                策略服务启动脚本
```

---

## 四、安全处理说明

`run_task3_tool_storage_scored.py` 的 docstring 里原本含有一处**队伍 token 的示例值**
（在「用法」示例的命令行中）。本提交包已将其替换为占位符 `<你的队伍token>`，
已扫描确认包内无残留。

源码中 `--team-token` 的参数默认值是 `None`，**不存在任何硬编码凭据**。
复现者需要向大赛主办方申请自己的 token。

---

## 五、复现所需但不在本包内的内容

以下由主办方提供或需自行准备，**不在本提交包内**：

1. **SouthGrid 仓库本体**（主办方提供，含 `orcalab_lerobot` 环境与仿真框架）
2. **OpenPI 仓库本体**（需按 `docs/openpi_deployment.md` 自行 `uv sync`）
3. **训练数据集 `g1_omnipicker_clear5_rand`**（100 集 / 132,288 帧 / 约 1.2 GB）
   —— 可按技术文档 §6.2 的命令自行采集，或向本队索取
4. **热启动 checkpoint**（`clear5_v1_l3090_lora/43500`，8.6 GB）
   —— 若从零复现，需先按 §4.2 所述训练固定顺序基线
5. **训练产出的 checkpoint**（`clear5_rand_lora_100ep/{5000,10000,11999}`，各 8.6 GB）

> 如果附件有大小限制，建议**优先上传技术文档与 `code/`**，
> 并在文档中说明大数据集/权重的获取方式。
