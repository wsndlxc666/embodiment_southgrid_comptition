# ============================================================================
# 需要加入 openpi 仓库 `src/openpi/training/config.py` 的两段代码
# ============================================================================
#
# 本文件不是可直接运行的脚本，而是**补丁内容**。复现时请把下面两段分别插入
# openpi 的 `src/openpi/training/config.py`：
#
#   ① `G1OmnipickerToolDataConfig` —— 插入到该文件已有的其它 `DataConfigFactory`
#      子类附近（本队放在 `RLDSDroidDataConfig` 之前）。
#   ② `pi05_g1_omnipicker_clear5_rand_lora` 那条 `TrainConfig` —— 插入到
#      `_CONFIGS` 列表里、已有的 clear5 系列条目附近。
#
# 另外还需要把 `pi05_g1_omnipicker_clear5_lora_s1500`（热启动源）那条 config
# 一并保留/加入，否则 `weight_loader` 指向的 checkpoint 无法对应。
#
# 依赖：本队对 openpi 的改动仅限 `config.py`（新增上述两段）与
# `src/openpi/policies/g1_omnipicker_policy.py`（数据变换，随交付提供）。
# 未修改任何模型结构或训练循环。
#
# ============================================================================

# ----------------------------------------------------------------------------
# ① DataConfigFactory：电力工具归纳数据集配置
# ----------------------------------------------------------------------------
class G1OmnipickerToolDataConfig(DataConfigFactory):
    """电力工具归纳数据集配置。"""
    repo_id: str = tyro.MISSING
    default_prompt: str = "整理工具"
    # 机器人的真实动作维度。模型内部固定用 32 维（pi05 基座权重是 32 维），
    # 数据在归一化后被 PadStatesAndActions 零填充到 32，模型输出时再截回此维数。
    action_dim: int = 18

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_head": "observation.images.cam_head",
                            "cam_wrist_r": "observation.images.cam_wrist_r",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 数据集里的动作键是单数 action，必须指定，否则默认读 "actions" 会报错
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[g1_omnipicker_policy.G1OmnipickerInputs(model_type=model_config.model_type)],
            outputs=[g1_omnipicker_policy.G1OmnipickerOutputs(action_dim=self.action_dim)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

# ----------------------------------------------------------------------------
# ② TrainConfig：随机顺序 clear5 微调（任务三本轮训练所用）
# ----------------------------------------------------------------------------
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_rand_lora",
        # 随机顺序 clear5(2026-09-23 采, 100 集 / 132,288 帧 / 18 维)：k∈[1,5]、
        # 顺序 60% 随机 + 20% 左到右 + 20% 右到左、位置/姿态/场上件数都随机。
        #
        # 这是**微调**不是重训，热启动源是固定左到右的 43500 基线：
        #   · 数据太小：batch4×12000 步 = 48k 样本 ÷ 132,288 帧 = 0.36 epoch，
        #     冷启动连"会抓这 5 把"都学不回来。
        #   · 43500 已经会全部技能(抓/运/入箱)，新数据新增的信息只是"顺序被打乱"。
        #   · 维度完全兼容：两边都是 18 维，action_dim=32 只是填充位；d19/d22 那几条
        #     要改 action_dim 是因为它们加了维，这条没加。
        #   · prompt 逐字相同(同样那 3 句)，文本侧无分布漂移。
        #
        # 超参照 d22idxR(同为"从已收敛 ckpt 微调")那一档，不照冷启动档：
        # 老 clear5_lora 的 peak 2e-4/warmup800 是冷启动用的，这里会训飞。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # 模型内部固定 32 维；真实动作 18 维由 padding+裁剪处理
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_rand",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,  # 3090 24G，与 orcalab 仿真共存 → 4 是实测安全值
        num_train_steps=12_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=12_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/clear5_v1_l3090_lora/43500/params"
        ),
        log_interval=100,
        # save_interval 与 keep_period **必须不同**：两者相等时 keep_period 的每个
        # 整数倍都被永久保留，等于一个都不删(2026-09-22 因此 ENOSPC 崩掉 5 小时训练)。
        # 单档 8.6G：1000/5000 下永久档只有 5000/10000 两档 ≈17G + 最新档。
        save_interval=1000,
        keep_period=5000,
    ),
