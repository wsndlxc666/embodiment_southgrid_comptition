# openpi 侧改动说明

本队对 [OpenPI](https://github.com/Physical-Intelligence/openpi) 的改动**仅限以下 4 个文件**，
没有修改模型结构、训练循环、数据加载器或推理服务。

复现方式：在干净 clone 的 openpi 上，用本目录下的文件覆盖同名文件即可。

| 本目录文件 | 覆盖到 openpi 的路径 | 改动性质 |
|---|---|---|
| `config_新增条目.py` | `src/openpi/training/config.py` | **新增条目**（不是整体覆盖，见下） |
| `g1_omnipicker_policy.py` | `src/openpi/policies/g1_omnipicker_policy.py` | 数据变换，新增文件 |
| `pi0_config.py` | `src/openpi/models/pi0_config.py` | 新增 5 个**带默认值**的字段 |
| `pi0.py` | `src/openpi/models/pi0.py` | 新增 3 个方法，收敛 loss 加权入口 |

---

## 1. `config_新增条目.py` —— 需要手工插入，不要整体覆盖

`config.py` 是 openpi 的核心配置文件，直接覆盖会丢掉上游的其它 config。
请把该文件里的两段代码**分别插入**：

- **① `G1OmnipickerToolDataConfig`**：插入到已有其它 `DataConfigFactory` 子类附近
- **② `pi05_g1_omnipicker_clear5_rand_lora` 那条 `TrainConfig`**：插入到 `_CONFIGS`
  列表里已有的 clear5 系列条目附近

另外，`weight_loader` 指向的热启动源
`checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/clear5_v1_l3090_lora/43500/params`
所对应的 config `pi05_g1_omnipicker_clear5_lora_s1500` 也必须存在于 `config.py` 中
（若从零复现，需先自行训练该基线）。

### 关于 `action_dim=32`

π0.5 基座权重的动作维度固定为 32。本任务的真实动作是 18 维，
数据经 `PadStatesAndActions` 零填充到 32 维送入模型，输出时由
`G1OmnipickerOutputs(action_dim=18)` 截回 18 维。**这不是改动，是官方既有的填充机制。**

---

## 2. `g1_omnipicker_policy.py`

两个 `DataTransformFn`：

- `G1OmnipickerInputs` —— `state` / `actions` 原样透传（不做额外投影）
- `G1OmnipickerOutputs(action_dim=18)` —— 把模型输出的 32 维截回 18 维

该文件在 `G1OmnipickerToolDataConfig.create()` 里被挂到 `data_transforms` 上。

---

## 3. `pi0_config.py` —— 5 个新增字段（全部带默认值）

```python
loss_weights: tuple[float, ...] | None = None   # 逐维 loss 权重，None = 全 1
grasp_frame_weight: float = 1.0                 # 抓取关键帧加权，1.0 = 关
grasp_grip_dim: int = 16                        # 右爪维度
grasp_close_thresh: float = 0.5                 # 判"闭合"的阈值
grasp_dims: tuple[int, ...] = (7, 8, 9, 18)     # 被加权的维度
```

**默认值均为「关闭」状态**，因此不设置这些字段时，行为与上游 openpi **逐位相同**。

> ⚠ 注意：**任务三本轮训练（`pi05_g1_omnipicker_clear5_rand_lora`）并未启用**
> `grasp_frame_weight`，也未启用 `loss_weights`。这两项是本队在另一条线上做的实验性
> 改动，此处一并提供仅为保证代码可编译、可复现。

---

## 4. `pi0.py` —— 3 个方法

把原本分散在两处的 loss 加权逻辑收敛成**唯一入口**：

```python
apply_loss_weights(per_dim)          # 逐维权重（归一化到均值 1）
apply_grasp_frame_weights(per_dim, actions)   # 按"真值动作块自己的 dim16"判闭爪帧
weight_loss_per_dim(per_dim, actions)         # 上述两者的唯一组合入口
```

`grasp_frame_weight <= 1` 时 `apply_grasp_frame_weights` 原样返回，
与不加这一项**逐位相同**。

---

## 5. 未改动的部分（澄清）

以下均**未修改**，沿用上游实现：

- 模型结构与前向计算（`pi0.py` 只加了 loss 加权辅助方法）
- 训练循环 `scripts/train.py`
- 推理服务 `scripts/serve_policy.py`
- 数据加载与归一化 `src/openpi/shared/normalize.py`
- `scripts/compute_norm_stats.py`

> 本队另外写了一个**独立的**统计量脚本 `../norm_stats_direct.py`，
> 它是新增文件、不覆盖官方脚本，只是绕开官方脚本的 HEVC 视频解码开销（见技术文档 §6.3）。
