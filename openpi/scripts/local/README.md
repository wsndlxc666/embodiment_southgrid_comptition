# scripts/local/ — 本机定制脚本

`openpi/scripts/` 下是本目录的**上游邻居**（openpi 自带）：`train.py`、`train_pytorch.py`、
`serve_policy.py`、`compute_norm_stats.py`。本目录放 G1 Omnipicker clear5 / d19 这条线上
自己写的加工、探针与启动脚本 —— 与上游分开，一眼能看出哪些是本地定制。

来源：原先混装在 `/home/dgl/lxc/data_maint/`（那个目录本意是"数据转移专用"，却塞进了脚本和日志）。
2026-09-13 迁入本目录 20 个。

## 数据集加工 / 统计

| 脚本 | 用途 |
|---|---|
| `build_d19_dataset.py` | **v1**：把 clear5 训练集从 18 维重建为 19 维（第 19 维 = 右臂末端到最近**工具**的欧氏距离），产出 `g1_omnipicker_clear5_d19_100ep`，并导出 `tool_refs.json` 给评测端 |
| `build_d19_goal_dataset.py` | **v2**：同上，但第 19 维改成到最近**关键点**的距离（参照点 = 5 个工具抓取点 ∪ 1 个盒子落点，取 min），产出 `g1_omnipicker_clear5_d19_goal_100ep`。修掉 v1 的两个洞：放盒瞬间读数 0.27 → 0.0097、每集归零 5 → 10 次。参照点用两遍扫描得出的**全局常量**，训练与推理共用同一组 |
| `analyze_cycle_phase.py` | 只读分析：核对 181 集的闭爪/开爪帧边界是否一致，并对比三种距离定义（min_k / 本周期工具 / 目标集）的连续性、取值分布、跳变 |
| `make_norm_stats.py` | 直接从 parquet 算 norm_stats，绕开 `compute_norm_stats.py` 的全量视频解码 |
| `norm_stats_fast.py` | 同上思路，但走官方完全相同的代码路径，只把视频解码跳过 |
| `compute_pooled_norm_stats.py` | 同上思路，pooled 读全部 parquet 的 state/action 列 |

> 这三个算 norm_stats 的脚本功能高度重叠，都源于"官方脚本要为每个样本解 20 步 × 2 路 HEVC 视频，太慢"。
> 留三份是因为各自验证角度不同（精确分位数 / 代码路径一致 / pooled），后续可考虑收敛成一个。

## 数据集体检 / 探针

| 脚本 | 用途 |
|---|---|
| `scan_clear5_train.py` | 轻量数值粗筛（零 sim，只读 parquet，~1-2 分钟）：结构、NaN/inf、动作极值离群、整集零方差 |
| `probe_action_channels.py` | 验证闭爪指令有没有活着走到训练目标 actions 里（走训练同一变换链，num_workers=0） |
| `probe_dataloader.py` | 纯数据侧探针：测取 N 个 batch 的耗时，不碰 GPU/模型 |
| `probe_policy_gripper.py` | 离线探针：拿"该闭爪"的真实帧喂策略，对照预测值 vs 真值，判定"不会闭爪"是训练侧还是评测侧 |
| `imgdiff_window.py` | 闭爪前后 cam_head 的逐帧图像差异，看有没有可用的视觉前兆 |

## serve 预热客户端

| 脚本 | 用途 |
|---|---|
| `warmup_clear5_8010.py` | 预热 clear5 serve(8010) 首次推理的 JAX JIT 编译 |
| `warmup_d19_8010.py` | 同上，针对 19 维 state；并顺带发一次 18 维做对照 |

## 训练 / 启动

| 脚本 | 机器 | 用途 |
|---|---|---|
| `train_local_3090.sh` | 本机 3090 | 训练执行器（nohup 后台） |
| `train_local_3090_jax.sh` | 本机 3090 | JAX-LoRA 训练器（pi05 基座加载 + 只训 LoRA） |
| `train_local_3090_jax_resume.sh` | 本机 3090 | JAX-LoRA 续训（不加 `--overwrite`） |
| `train_d19_resume.sh` | 本机 3090 | d19 专属 config/exp 续训，**绝不加 `--overwrite`**（那句 "Wiped checkpoint directory" 会清空全部存档） |
| `run_clear5.sh` | DGX k24 | 训练启动 + 实时查看（tmux 版，SSH 断开照跑） |
| `train_launch.sh` | DGX k24 | 实际训练执行器（在 tmux 会话内运行） |
| `prune_clear5_ckpts.sh` | 本机 | ckpt 瘦身：只在 10000 整数倍留参考档 + 保留最新完整档 |
| `stop_at_200.sh` | 本机 | 落盘总数到 200 即停采（watchdog + collector），供本机训练腾 GPU |
| `worker_sweep.sh` | DGX k24 | 扫 num_workers 稳定性/速度 |

## 日志去向

原名在 `data_maint/` 的 4 个脚本把日志写死到 `data_maint`，搬迁时已改为
**`/home/dgl/lxc/.runlogs/`**，避免 `data_maint` 继续堆积训练日志：

- `train_local_3090.sh` → `.runlogs/train_${EXP}.log`
- `train_local_3090_jax.sh` → `.runlogs/train_jax_${EXP}.log`
- `train_local_3090_jax_resume.sh` → `.runlogs/train_jax_${EXP}_resume.log`
- `train_d19_resume.sh` → `.runlogs/train_d19_100ep.log`

历史日志仍留在 `data_maint/*.log`（含 635K 的 `train_d19_100ep.log`）。

## 运行方式

脚本内全部是绝对路径，没有 `__file__` / 相对路径依赖，所以从哪调用都行。但涉及
`import openpi.*` 的探针脚本建议在 openpi 根目录下跑：

```bash
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
python scripts/local/scan_clear5_train.py
```

## 留在 data_maint 的是什么

`/home/dgl/lxc/data_maint/` 现在只剩**数据转移**相关：`build_train_from_landing.py`
（从采集落盘 repo 重建 train 快照）、`merge_legacy_into_landing.py`（把 legacy 20 集并入落盘 repo）、
`README_迁移落位.md`（迁移说明）、目标机 k24 的环境构建脚本三个，以及历史日志。
