"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.g1_omnipicker_policy as g1_omnipicker_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # ---- 周期相位重加权 / 反事实增强（见 openpi/training/cycle_index.py）----
    # 只对"action 里带 cycle_idx(第 21 维)"的数据集有意义。三个都默认关：
    #   cycle_boundary_window = 0 时整条链路完全不参与，行为与改动前逐字节一致。
    # 生效范围 = 每次 cycle_idx 变化点之后的这些帧。
    cycle_boundary_window: int = 0
    # 这些帧的过采样倍数（1.0 = 不过采样；10.0 = 每轮出现 10 次）。只改 batch 成分，
    # 不改 num_train_steps。仅支持单卡；DDP 下会报错而不是静默失效。
    cycle_boundary_weight: float = 1.0
    # 以该概率把边界帧观测里的 goal_y/cycle_dist 换成**同集另一相位**的真实值
    # （cycle_idx 保持正确），逼模型从 cycle_idx 拿相位。0.0 = 关。
    cycle_counterfactual_prob: float = 0.0
    # 命中时**连两路图像一起**换成供体（异相位、同相位内偏移）那一帧的，只有 cycle_idx
    # 和 action 真值来自参照帧。只换 dims19/20 已实测无效（模型照抄图像，梯度仍是 0），
    # 必须换图才能让 idx 成为相位身份的唯一解释。默认 False 保留旧行为可复现。
    cycle_counterfactual_images: bool = False
    # 按**相位**过采样，形如 ((0, 5.0), (1, 5.0), (4, 3.0))：相位 p 的帧每轮出现 round(w) 次。
    # 相位 p = 第 p 件工具被取走的整轮（接近→下探→闭爪→抬起→搬运→放下），p 从 0 起、
    # 顺序就是采集顺序 扳手/螺丝刀/电工刀(左)/手电筒/电工刀(右)。所以 p=0 那一轮是
    # "五件工具都在场"的唯一一段。空元组 = 关，与改动前逐字节一致。
    # 与 cycle_boundary_weight **相乘**而不是二选一：边界帧同时是相位开头，会同时命中。
    cycle_phase_weights: tuple[tuple[int, float], ...] = ()

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
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

@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # 若设置,每个 log_interval 把逐维 loss 追加写进这个 jsonl(每行一个 step)。
    # 用来盯"爪子维/距离维到底有没有在学",而不是只看一个被平均掉的 loss。
    diag_log_path: str | None = None

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_tool_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # 必须匹配 pi05 基座权重（32 维），机器人真实动作 18 维由 padding+裁剪处理
            action_horizon=20,  # 50 步太重(30h+)；20步=1s 重规划，提速约2x
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_tool",
            # 关键：数据集的 task 即工具名，按任务注入 prompt，模型才能学会「按工具名取工具」
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,  # 3090 24G 装不下 32，降到 8（留足余量）
        num_train_steps=12_000,  # 250集数据；3.8s/步 ≈ 12.7h。评测不够再加，存档每500步都有
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=12_000,  # 对齐 num_train_steps，让学习率完整退火到 2e-5
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        log_interval=100,
        save_interval=500,  # 每500步(~20min)存一次，中断最多丢20分钟；12000步共24个存档
        keep_period=500,  # 关键: orbax 默认 max_to_keep=1 只留最新一个；keep_period=500 把每个500步的存档永久保留, 供之后选最优
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_single_grasp_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # 单工具任意位姿抓取数据集：task 即抓取指令（如“抓取扳手”），prompt 逐集注入
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_single_grasp",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,  # 3090 24G，且与 orcalab 仿真(约5.3GB)共存 → 降至4防OOM
        num_train_steps=8_000,  # 4×8000=32k 窗口样本≈1 epoch; 实测步速后再调
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=800,
            peak_lr=2e-4,
            decay_steps=8_000,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        log_interval=100,
        save_interval=500,
        keep_period=500,
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_single_place_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # 完整 pick+place 数据集（任意位姿抓起→放入工具箱）：在"任意位姿抓取"模型
        # (g1_single_grasp_v1/7999) 之上继续 LoRA 微调，把后半段续成 搬运→入箱→松开。
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_single_place",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,  # 与 orcalab 仿真共存防 OOM（同 stage A）
        num_train_steps=2_000,  # 首批 20 集数据(约10k帧)，先训 2000 步看效果；不足可续训
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2e-4,
            decay_steps=2_000,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_g1_omnipicker_single_grasp_lora/g1_single_grasp_v1/7999/params"
        ),
        log_interval=100,
        save_interval=500,
        keep_period=500,
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # 模型内部固定 32 维；真实动作 18 维由 padding+裁剪处理
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # Clear-All 数据集：固定 5 把全在场、不点名、固定从左到右一集全收，恒定全收指令。
        # 指令是"一整句全收"(无工具名)，按任务注入 prompt → 模型把指令映射到
        # "视觉识别 → 从左到右清空条带"。数据全部冷启动新采，故从 pi05_base 起训、
        # norm_stats 用 clear5 自己的。训练用干净快照 train(剔 legacy)；落盘全量在 clear5。
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_train",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,  # 3090 24G，且与 orcalab 仿真共存 → 降至4防OOM
        num_train_steps=12_000,  # 首块 ~40 集(每集~2100帧≈5把)先训 12k 自测；数据续采后按 epoch 再调
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=800,
            peak_lr=2e-4,
            decay_steps=12_000,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        log_interval=100,
        save_interval=500,
        keep_period=500,
    ),
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
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_lora_s1500",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # 模型内部固定 32 维；真实动作 18 维由 padding+裁剪处理
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # 本机 3090 首轮快跑(2026-09-08)：train 快照 181 集。1500 步 + 短 LR 日程
        # (warmup120/decay1500 匹配短跑，原 warmup800 会让 1500 步全程在预热而失真)。
        # batch=4 为 3090 24G 实测安全值(batch8 首步 OOM ~22.6GB>可用 23.2GB)；
        # 与 orcalab 共存时更低，DGX(更大显存)可 CLI 上调。验证后再用完整配置另起满训。
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_train",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,
        num_train_steps=1_500,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=120,
            peak_lr=2e-4,
            decay_steps=1_500,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        log_interval=10,
        save_interval=500,
        keep_period=500,
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d19_lora_20k",
        # 19 维版本(2026-09-10 起)：真实动作 18 维 + 第19维「右臂末端到最近工具的距离」。
        # 目的:给"接近→抓取→搬运→松开"这条相位结构一个显式的、可学的信号。
        #   距离维在数据里是锯齿波:接近某件工具时单调减小到 0,抓起来走开时增大。
        # 模型内部仍固定 32 维(pi05 基座权重是 32 维),19 维经 padding 到 32,
        # 所以 action_dim 用 19 只影响输出裁剪,模型结构完全不变。
        #
        # 损失：距离维和两个右爪维各给 5 倍权重。理由是类不平衡——
        # 181 集里"当前爪子是开的"的样本中只有 11.43% 在未来 1 秒内需要闭合，
        # 不加权时"该闭爪"的梯度会被 88.57% 的"保持张开"淹没。
        # 权重会被归一化成均值 1，所以 loss 绝对量级与历史 run 可比。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 注意：逐维 loss 的长度是"填充后"的 action_dim=32（真实的 19 维后面
            # 补了 13 个零），不是数据集的 19 维。所以是 16 + 3 + 13。
            # 16/17 = 右爪内/外，18 = 距离，19..31 = 填充维(权重保持 1，别让它们
            # 影响归一化后的相对比例)。
            loss_weights=tuple([1.0] * 16 + [5.0, 5.0, 5.0] + [1.0] * 13),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d19_100ep",
            action_dim=19,
            base_config=DataConfig(prompt_from_task=True),
        ),
        # 从 43500 热启动(维度完全兼容:两边都是 32 维模型)。不是从 pi05_base 重来，
        # 因为 20000 步从零训大概率还到不了现有水平。相当于"换数据配方继续训"。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/clear5_v1_l3090_lora/43500/params"
        ),
        exp_name="clear5_d19_lora_100ep",
        batch_size=4,
        num_train_steps=20_000,
        # 热启动用较低峰值 LR(1e-4 而非原 2e-4)，避免把已学到的表征冲掉。
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        # 单档 8.6G：每 1000 步存，只留 5000 的倍数+最新，峰值约 5 档 ≈ 43G
        save_interval=1_000,
        keep_period=5_000,
        # 逐维 loss 写到新 data 目录下(不动原始数据集)
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d19_100ep/train_diag.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d19goal_lora_20k",
        # 19 维版本 v2 (2026-09-13)：把第19维的"参考点"从"5 个工具抓取点"扩成
        # "5 个工具抓取点 ∪ 1 个盒子落点"(共 6 个), 仍然取 min。
        #
        # 为什么改：v1 只在工具处归零, 把工具放进盒子那一刻的读数约 0.27 m ——
        # 手明明已经在盒子里了, 数字却说"离最近的目标很远", 且搬运途中这个数只反映
        # "盒子碰巧离哪把工具最近", 没有语义。v2 让"抓到手"和"放进盒"两个关键
        # 时刻都归零, 每集归零次数 5 -> 10, 相位信息翻倍。
        # 实测(100集, 第19维)：
        #   近(d<0.05) 18.6% -> 33.8%;  远(d>0.20) 73.3% -> 38.4%
        #   中位 0.2728 -> 0.1247;      最大相邻帧跳变 0.0850 -> 0.0728
        #   放盒帧读数 0.2716 -> 0.0097
        #
        # 参考点用**跨集全局常量**(scripts/local/build_d19_goal_dataset.py 两遍扫描
        # 得出, 落在 tool_refs.json), 不用每集自己的实测点。代价是闭爪帧读数不再是
        # 精确 0 而是 ~1.1 cm(p95 2.3cm), 换来训练与推理完全同一个函数 ——
        # 推理端只能拿一组固定常量, 用每集实测点反而会制造训练/推理错配。
        # 信号/噪声比 25:1。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 逐维 loss 长度是"填充后"的 action_dim=32(真实 19 维后补 13 个零)。
            #   0..6   左臂(pos3+quat4)  -> 0.05  压到极低，理由见下
            #   7..13  右臂
            #   14..15 左爪
            #   16..17 右爪内/外
            #   18     目标点距离
            #   19..31 填充维(目标恒 0)
            #
            # 为什么把左臂压到 0.05：分位数归一化是**尺度无关**的——每维都除以自己的
            # 分位跨度，于是"在只动 4.6mm 的关节上错 0.79mm"和"在扫过 731mm 的右臂上
            # 错 31mm"被打成同一量级的分数。实测(v1 稳态 step12300-13800)：
            #     左臂 物理误差 0.09~0.79 mm  ->  占损失质量 44.8%   ← 早已控制到亚毫米
            #     右臂 物理误差 12.8~31.4 mm  ->  占损失质量  6.8%   ← 真正要学的
            #     右爪                        ->  占损失质量 43.2%
            # 而且左臂在数据里几乎静止(位置逐维标准差 0.0007/0.0005/0.0012 m，左爪恒
            # 0.0480)，任务上完全无关。它白吃掉近一半梯度，压掉后把这部分让给右臂+右爪。
            # 不取 0 是想留一个极弱的锚，防止共享主干为右臂更新时把左臂输出带偏。
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [5.0, 5.0, 5.0] + [1.0] * 13),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d19_goal_100ep",
            action_dim=19,
            base_config=DataConfig(prompt_from_task=True),
        ),
        # 与 d19 v1 同样的热启动点：两边模型都是 32 维, 权重完全兼容。
        # 想观察 v2 距离维的净效果, 从 43500 起最干净(换了数据配方, 不换起点)。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_lora_s1500/clear5_v1_l3090_lora/43500/params"
        ),
        exp_name="clear5_d19goal_lora_100ep",
        # 让策略服务在 websocket 握手里 advertise state 维度, 评测脚本据此**硬校验**。
        # 没有这个的话"推理端少发几维"是完全静默的 —— pi0.5 把 state 离散成 prompt 里的
        # "State: v0 ... vN;" 文字, 少一维只是那条串短了个数, 模型看到从没见过的输入,
        # 不报任何错, 只是分数莫名其妙变差。d19 那次就栽在这上面。
        policy_metadata={"state_dim": 19, "state_names_tail": ["goal_dist"]},
        batch_size=4,
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        # 逐维 loss 单独写一个文件, 方便和 v1 的 train_diag.jsonl 并排对比
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d19_goal_100ep/train_diag.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22_lora_20k",
        # 22 维 v3「周期感知」版 —— 在 d19_goal(v2) 的 19 维后面再加 3 维:
        #   19 goal_y      当前目标工具在 base 系下的 y
        #   20 cycle_dist  ‖ee − 当前目标工具‖, 在每轮的闭爪帧上归零 → 那个 0 就是"合爪"
        #   21 cycle_idx   1..5 = 正在收第几把; 6 = 五把都放完, 可以停
        # 数据: scripts/local/build_d22_cycle_dataset.py (前 19 维与 v2 逐位相同, 已校验 0.00e+00)
        #
        # 为什么只加 y 不加 xyz: 实测 5 把工具在 base 系下 ——
        #   y 跨度 407.9mm, 相邻差 77~131mm  → 5~10 倍于布局抖动, 唯一可靠判据
        #   x 跨度  44.7mm, 相邻差 11.5~33.1mm → 落在抖动带(±8mm, 12%概率±40mm)内, 读不出身份
        #   z 跨度  19.8mm, 相邻差  3.5~17.1mm → 就是工具厚度, 且跨度和左臂那几维
        #                                        (1.4~18mm, 曾吃掉 63.8% loss)同一区间
        # 即 x 是噪声维、z 是小跨度陷阱维, 加进去只有害处。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 逐维 loss 长度是"填充后"的 action_dim=32(真实 22 维后补 10 个零)。
            #   0..6   左臂(pos3+quat4)                      -> 0.05
            #   7..15  右臂(pos3+quat4) + 左爪2
            #   16..18 右爪内/外 + goal_dist                 -> 5.0
            #   19..21 **新增** goal_y / cycle_dist / cycle_idx -> 1.0
            #   22..31 填充维(目标恒 0)
            # 前 19 个权重**逐字沿用 d19goal 那档**, 热启动后对老维度的重视程度不变,
            # 只有新增 3 维是新的。新 3 维取 1.0(中性): 它们在推理时会被截掉,
            # 只当辅助训练信号用, 不值得再加权。
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [5.0] * 3 + [1.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22_100ep",
            action_dim=22,
            base_config=DataConfig(prompt_from_task=True),
        ),
        # 热启动: 从**它自己的** 9000 步接上, 不是从基座重来。
        # 架构完全兼容 —— pi0.5 的 state 是**文字**不是张量, 加 state 维只改 prompt 字符串,
        # 不动任何权重形状(action_dim 内部恒为 32), 所以 CheckpointWeightLoader 直接吃。
        # 新的 norm_stats 会由 22 维数据在新 ckpt 的 assets/ 下现算。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d19goal_lora_20k/clear5_d19goal_lora_100ep/9000/params"
        ),
        # exp_name 与上面 d19goal 那档不同名 → ckpt 落在**新的**
        # checkpoints/pi05_g1_omnipicker_clear5_d22_lora_20k/ 下, 不会覆盖老结果。
        exp_name="clear5_d22_lora_100ep",
        # 同 d19goal: 把 state 维度写进握手 metadata, 评测脚本据此硬校验。
        # 22 = 19(含 goal_dist) + goal_y / cycle_dist / cycle_idx。
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
        # 与 d19goal 同配方(warmup 500 / peak 1e-4 / 20k 余弦)以便并排比较。
        # 注意这是"重新热身"而不是"续训降 LR": 模型会被 1e-4 再扰动一次。
        # 想更保守把 peak_lr 降到 2e-5(见 memory: 长尾平台那次用的是 2e-5)。
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        # 每 1000 步存一次 —— d19goal 那次 loss 在 step 1000 就平了,
        # 这次主要看**新 3 维有没有被用起来**(翻 train_diag.jsonl 的 dim19/20/21),
        # 不一定要跑满 20k, 随时可以从最近的 ckpt 停。
        save_interval=1_000,
        keep_period=5_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22_100ep/train_diag.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22idx_lora_20k",
        # 在 d22 之上**只加数据侧干预**，专门把 cycle_idx 的指向训出来。
        # 与 d22 那档的差异只有三处（其余逐字相同，方便并排归因）：
        #   1. 热启动换成 d22 自己的 19000（d22 那档是从 d19goal 9000 起的）
        #   2. data 上开 cycle_boundary_window=60 / weight=10 / counterfactual_prob=0.5
        #   3. exp_name / diag_log_path 换新，不动老结果
        #
        # 依据（全部实测，见 scripts/local/phase_target_conflict.py 等四个脚本）：
        #   变化点(366/766/1166/1566)之后 40 帧内，手臂位置云心彼此只差 8~18mm 而各自散布
        #   1~46mm；65~68% 的样本能在**别的相位**里找到位置差<10mm 的邻居，而这些样本未来
        #   20 帧的真值轨迹相差中位 44mm(p90 76mm)，同相位抖动地板只有 6mm —— 差 7~10 倍。
        #   这段窗口只占全集 14.3%，且模型还能从 dim19 goal_y(逐集 std=0.0000 的纯净相位
        #   标签)或头相机抄近道，所以 cycle_idx 实测只拿到 0.09 的权柄。
        #   过采样把这段窗口的占比抬到约 60%，反事实增强打断 goal_y/cycle_dist 这条近道，
        #   只留 cycle_idx 可读 ⇒ action 的 dim19/20 也就只剩"从 cycle_idx 反推"一条路。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 逐维权重**逐字沿用 d22**：这一轮只动数据不动 loss，否则分不清是哪个起的效。
            # （想单独试"加大索引权重"就把 19..21 那三个 1.0 改成 3.0，作为下一个对照。）
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [5.0] * 3 + [1.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22_100ep",
            action_dim=22,
            # 归一化显式指向 d22 那一档的**同一份** norm_stats。
            # 默认 assets_base_dir 是 ./assets/<config name>/，本档名字不同 → 找不到 stats
            # 就直接报 "Normalization stats not found"。更重要的是：这一轮只动采样/增强，
            # 归一化必须逐位相同，否则 dim19/20/21 的标准化尺度一变就没法归因。
            assets=AssetsConfig(
                assets_dir=str(pathlib.Path("./assets/pi05_g1_omnipicker_clear5_d22_lora_20k").resolve())
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                # 变化点后 60 帧：实测 k=40 歧义最重，k=60 仍有 41% 样本的异相位邻居在
                # 10mm 内，k=100 才干净分开。
                cycle_boundary_window=60,
                # 边界窗口占全集 14.3%，x10 之后每个 batch 里约 60% 是边界样本。
                cycle_boundary_weight=10.0,
                # 只打断一半。留一半是刻意的：评测时 goal_y 是环境给的**正确**值，
                # 不能把模型训成彻底不信它。
                cycle_counterfactual_prob=0.5,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d22_lora_20k/clear5_d22_lora_100ep/19000/params"
        ),
        exp_name="clear5_d22idx_lora_100ep_pilot3k",
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
        # 先跑 3000 步量指标（3090 上约 1.5~2 小时），好再决定跑不跑满。
        # 3k 内走完整余弦，所以这是"短跑"，不是"20k 的前 15%"。
        num_train_steps=3_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=1e-4,
            decay_steps=3_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22_100ep/train_diag_idx.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22idxI_lora_20k",
        # 改法 A：在 d22idx 之上**再换图像**，让 cycle_idx 成为相位身份的唯一解释。
        #
        # 为什么非换不可（实测）：d22idx 那轮只把 dims19/20 换成异相位真值、图像不动，
        # 训完 action dim19 的 loss 掉 191 倍、dim21 掉 108 倍 —— 模型确实在 index 通道
        # 内部解开了 (19,20)↔21 的矛盾，**但手臂没跟着 idx 走**。因为"读图"和"读 idx"
        # 给出同一个 action，动作空间里没有哪个分量是只有 idx 能解释的
        # ⇒ 指向 idx 的**相对梯度恒为 0**，再多的步数也训不出来。2999 步的结论不是
        # "欠拟合"，是"这个信号根本不存在"。所以这一轮把图像也换掉，凭空造出那条梯度。
        #
        # 与 d22idx（pilot3k）的差异，**只有两处**（其余逐字相同）：
        #   1. data 上多开 cycle_counterfactual_images=True ← 本轮的主角
        #   2. loss_weights 的 19/20/21 由 1.0 → 3.0        ← 用户要的"加大索引权重"
        # 注意这两处是**一起动的**，所以本轮的 A/B 对照只能回答"这套组合有没有用"，
        # 不能把功劳单独归给哪一个。想归因就再跑一轮只开换图、权重留 1.0 的短跑。
        # 热启动源与 pilot 相同（d22 自己的 19000），所以 pilot 就是这一轮的对照组。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # 归一化里 loss_weights 会 **除以自己的均值**（见 models/pi0.py），所以
            # 改三个数会顺带整体缩放。实算：19..21 相对权重 0.857 → 2.215（x2.59），
            # 其余 1.167 → 1.354（x0.86）。是真正的相对重加权，不是全局放大。
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [5.0] * 3 + [3.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22_100ep",
            action_dim=22,
            # 与 d22idx 逐位相同的 norm_stats，否则 dim19/20/21 的标准化尺度一变就没法归因
            assets=AssetsConfig(
                assets_dir=str(pathlib.Path("./assets/pi05_g1_omnipicker_clear5_d22_lora_20k").resolve())
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                cycle_boundary_window=60,
                cycle_boundary_weight=10.0,
                # 只打断一半。留一半是刻意的：评测时 goal_y 是环境给的**正确**值，
                # 不能把模型训成彻底不信它；也不能训成彻底不信视觉（抓取末段还要靠它）。
                cycle_counterfactual_prob=0.5,
                # ← 本轮的新东西
                cycle_counterfactual_images=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d22_lora_20k/clear5_d22_lora_100ep/19000/params"
        ),
        exp_name="clear5_d22idxI_lora_100ep",
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
        # 2999 步被证明测的是"信号不存在"，不是"量不够"。既然这一轮换的是信号本身，
        # 就一次跑满 20k（3090 上约 10~12 小时），中途每 1000 步存一次随时可停。
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22_100ep/train_diag_idxI.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22idxP_lora_20k",
        # 相位加权：把"第 1、2、5 件工具"那三轮的帧按倍数过采样（用户 2026-09-19 的要求）。
        #
        # 起因（闭环实测，.runlogs/rollout_d22idxI_14000.jsonl 81 块全 20 帧）：
        #   模型能完整做对 2/5 —— 手电筒(第4件)与电工刀(左)(第3件)，下探/闭爪/搬运/开爪
        #   全对；但它**从第 0 块就往手电筒去**，整轮没碰过扳手/螺丝刀/电工刀(右)，最后
        #   还回到已掏空的手电筒槽位再抓一次空气（所以是 2/5 不是 3/5）。是**相位/次序**
        #   错，不是抓取技能缺 —— 加权就是针对这个。
        #
        # 为什么是 0:3 而不是"想加多大加多大"（scripts/local/verify_phase_weight.py 实测）：
        #   现状（cycle_boundary_weight=10）下每轮的相位成分是
        #       p0 7.6% | p1..p4 各 19.6% | p5 14.1%
        #   相位 0 是六个里**最低**的。原因是边界过采样只落在相位 1..5 的开头（帧 0 那段
        #   home 位姿被 include_episode_start=False 刻意排除），于是"必须第一个做的那件
        #   工具"反而样本最少，赤字 2.6x。所以 0:3 首先是**把这个无意造成的赤字补平**
        #   （→17.3%），其次才是额外强化；1:2/4:2 是用户要的另外两件。
        #   代价：p2/p3（当前唯一做对的两件）从 19.6% 降到 14.8%。这一行是本档最大的
        #   风险 —— 拿已经会的换还不会的。想更保守就整组降到 (0,2)(1,2)(4,2)。
        #
        # 对照关系：**只动采样器**，其余（模型/loss_weights/反事实/热启动源）与 d22idxI
        # 逐字相同 ⇒ 任何差异都可归给采样成分，这是 memory 里反复强调的"唯一合法对照"。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [5.0] * 3 + [3.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22_100ep",
            action_dim=22,
            assets=AssetsConfig(
                assets_dir=str(pathlib.Path("./assets/pi05_g1_omnipicker_clear5_d22_lora_20k").resolve())
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                cycle_boundary_window=60,
                cycle_boundary_weight=10.0,
                cycle_counterfactual_prob=0.5,
                cycle_counterfactual_images=True,
                # ← 本轮的主角。相位 p = 第 p+1 件工具被取走的整轮（0 基）。
                # 倍数与边界倍数是**相乘**关系，两者作用在不同帧集合上，不必二选一。
                cycle_phase_weights=((0, 3.0), (1, 2.0), (4, 2.0)),
            ),
        ),
        # 热启动 = 改法 A 跑到的最强存档。**注意**：14000 是"最新那个"而不是 5000 倍数，
        # 若之后有人 resume 了 d22idxI，orbax 会把它删掉，本档的初始化源就没了
        # （数据 14000 = 19000 + 14000 步的成熟度；5000/10000 也在，可换）。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d22idxI_lora_20k/clear5_d22idxI_lora_100ep/14000/params"
        ),
        exp_name="clear5_d22idxP_lora_100ep",
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
        # 相位加权把每轮样本从 480k 抬到 633k，但 num_train_steps 是**步数**不是 epoch 数，
        # 所以实际见过的样本量不变 —— 变的是每个 batch 的成分。要跑到跟 d22idxI 同样的
        # "见过多少遍数据"就得按 633/480 = 1.32x 加步数，这里不加：本轮比的是成分不是量。
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22_100ep/train_diag_idxP.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22idxR_lora_12k",
        # 接 idxP 往下推。**闭环实测**（.runlogs/rollout_idxP_3000.jsonl，runner 的 summary）：
        #   {"cycle_idx_end": 2, "placed": {"扳手": true, 其余 false}, "elapsed_total": 165.0}
        # idxP 相比 idxI 的最大进步是**次序对了** —— 它开始按 扳手→螺丝刀→电工刀(左) 的顺序
        # 去抓前三件（idxI 是第 0 块就往手电筒去）。但两件都**没入箱**，用户观察是"抓掉了"。
        #
        # 两件事其实是**同一个根因**：右爪抓不牢 → 工具中途掉了 → 箱内没有"开爪释放"沿
        # → runner 的 cycle_idx 卡在 2 推不动（cycle_trace 只有 1→2 一次跳变）→ 第 4 下
        # 又回到刚做过的那件。所以本轮**先修抓取**，相位权重只是辅助。
        #
        # 为什么锁定在右爪（norm_stats 实测，不是推测）：
        #   L爪14/L爪15  q01=q99=0.0480  std=0.0000  ← **100 集里全程没动过**（恒=张开）
        #   L臂1..7      std 0.0008~0.0048           ← 几乎静止
        #   R爪16/R爪17  q01=0.0480 q99=0.9998 span 0.9518  ← 只有它在开合
        # ⇒ 整个任务只用**右臂+右爪**完成（clear_eval 转储里那个 freeze_left:true 同义）。
        #   左臂 0.05 / 左爪 1.0 的低权重是**对的**，别去"修正"它。
        #   训练日志里右爪也是 loss 最大最尖的两维（冲到 0.178，其余全在 0.01 量级）：
        #     Step 5180: 右爪16=0.11256 右爪17=0.11005 距离18=0.02633 左爪14=0.00353
        #   （左爪 loss 低不是因为学得好，是因为它**不动** —— 见 memory
        #    openpi-quantile-norm-scale-free 那条"近静止维度 loss 天然低"的陷阱。）
        #
        # 本轮相对 idxP 只有**两处**改动，各自对应一个症状：
        #   [改动1 抓取] 右爪 16/17 权重 5.0 → 8.0。占 loss 权重比 23.1% → 32.4%。
        #                直接治"掉件"。距离 18 保持 5.0 不动（它是另一个失败源，别一起动）。
        #   [改动2 次序] 相位权重补上 p2/p3。idxP 是 ((0,3),(1,2),(4,2))，实测成分
        #                p0 17.3 | p1 21.2 | p2 14.8 | p3 14.8 | p4 21.2 | p5 10.7
        #                —— p2/p3 是六个里最低的，而模型恰好卡在**进入 p3（手电筒）**。
        #                补成 (2,2)(3,2) 后 p1..p4 齐平在 ~21%。
        #                已知代价：p5（电工刀(右)）成了新的最低（~10.7%）。但模型现在根本
        #                走不到 p5，它不是当前的约束；等 p3/p4 做出来之后下一轮必须补 p5。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # [改动1] 右爪 16/17: 5.0 → 8.0
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22_100ep",
            action_dim=22,
            assets=AssetsConfig(
                assets_dir=str(pathlib.Path("./assets/pi05_g1_omnipicker_clear5_d22_lora_20k").resolve())
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                cycle_boundary_window=60,
                cycle_boundary_weight=10.0,
                cycle_counterfactual_prob=0.5,
                cycle_counterfactual_images=True,
                # [改动2] 补上 p2/p3
                cycle_phase_weights=((0, 3.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0)),
            ),
        ),
        # 热启动 = idxP 的 5000（正式档，非滚动档）。**不要**用 _keep_idxP/3000：那是用户
        # 评过的档，本轮要接的是"训到 5000"的状态。idxP 只跑到 5410 就被叫停了，
        # 所以 5000 就是它的成熟度上限。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d22idxP_lora_20k/clear5_d22idxP_lora_100ep/5000/params"
        ),
        exp_name="clear5_d22idxR_lora_100ep",
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
        # 12000 步 ≈ 5.3 小时（BFC 预分配实测 1.6 s/step）。idxP 在 5410 步就已经从 2/5 的
        # 次序错变成"前三件按序去抓"，说明这个档的收益在 5k~12k 就已经显形；再往上加步数
        # 之前，先把抓取这个瓶颈解决掉更值。
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
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22_100ep/train_diag_idxR.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_d22idxS_lora_12k",
        # 接 idxR/11999 往下推。**相对 idxR 只换数据源**（recipe 逐字不变），
        # 所以结果可以单变量归因到"相位段重排"这件事上。
        #
        # 为什么换数据（2026-09-20 实测坐实的根因）：
        #   原数据 100/100 集的 cycle_idx 跳变帧完全相同（366/766/1166/1566/1966），
        #   每集都是 2101 帧 ⇒ **cycle_idx 在训练里是帧号的纯函数**，它携带的信息
        #   不超过"图像进度"本身。于是模型学成"按视觉进度走一张时间表"，而不是
        #   "读相位决定去拿哪件"。证据链：
        #     · 训练数据里 idx 的跳变帧恰好是右爪『闭→开』边沿的下一帧
        #       （8/8 集、5/5 个边界，差恒为 +1）—— 相位在训练中的唯一含义是"刚松手"。
        #     · 闭环实测（.runlogs/rollout_idxR_notl.log）：idx 一次都没超过 3，
        #       从 t=85.1s 到 t=292.0s 冻在 3 长达 207s，而模型**一直在推理**
        #       （step 单调增、每 3.2s 恰好 +120 = 6 次推理、动作块每块都在变）。
        #       即：不是推理停了，是图像与推进规则互相矛盾时模型收到了训练里
        #       从未出现过的 state，行为退化成了重复微指令（37% 的块位移 <1mm）。
        #
        # 数据怎么造（scripts/local/build_phase_shuffled_dataset.py）：
        #   每集按 24 种排列轮换重排 seg1..seg4（seg0 恒第一、seg5 恒收尾，
        #   因为只有 seg1..seg4 两端都落在箱位，互换才不产生大跳变）。
        #   **只改 dim21 一维**，其余 21 维逐帧逐字节等于源 —— 工具没换，
        #   只是位次变了，所以 dim19 goal_y 仍然指向真正要拿的那件工具。
        #   效果：idx 不再能通过固定表推出 goal_y（同一个 idx 在不同集里对应
        #   不同工具），"位置 p ⇒ 工具 p" 这条捷径被拆掉。
        #
        # norm_stats 直接复用**源数据集那一份**（asset_id 显式指向 d22_100ep）：
        #   `asset_id` 不给就默认取 repo_id，新 repo_id 找不到 → norm_stats=None →
        #   `data_loader.transform_dataset` 直接抛
        #   "Normalization stats not found. Make sure to run scripts/compute_norm_stats.py"。
        #   而这份 stats 对重排后的数据是**精确正确**的，不是将就：实测用
        #   `openpi.shared.normalize.RunningStats` 把两个数据集各扫一遍逐维对比，
        #   mean/std/q01/q99 全维最大差 **3.2e-13**（纯浮点累加噪声）——
        #   因为每集各段长度不变（1×366 / 2..5×400 / 6×135），重排是双射，
        #   dim19/20/21 的边缘直方图一格没动。顺带这也保住了"只换数据源"的单变量性。
        #
        # ⚠ 本轮**真正的第一手修法其实在 harness**：--progress_gate 默认已改回
        #   grip（与训练语义逐帧对齐）。早上应该**先用现有 idxR/11999 + 修好的
        #   harness** 跑一次闭环 —— 那个不需要等训练。本档是它的后备。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            loss_weights=tuple([0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10),
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_d22idxS_100ep",
            action_dim=22,
            assets=AssetsConfig(
                assets_dir=str(pathlib.Path("./assets/pi05_g1_omnipicker_clear5_d22_lora_20k").resolve()),
                # 必须显式给：不给就默认取 repo_id（=...d22idxS_100ep），那个目录不存在
                # ⇒ norm_stats=None ⇒ 起训即 ValueError（冒烟实测）。
                # 复用源那一份是**精确正确**的，见上面那段的实测数据。
                asset_id="g1_omnipicker_clear5_d22_100ep",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                cycle_boundary_window=60,
                cycle_boundary_weight=10.0,
                cycle_counterfactual_prob=0.5,
                cycle_counterfactual_images=True,
                cycle_phase_weights=((0, 3.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0)),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_g1_omnipicker_clear5_d22idxR_lora_12k/clear5_d22idxR_lora_100ep/11999/params"
        ),
        exp_name="clear5_d22idxS_lora_100ep",
        policy_metadata={
            "state_dim": 22,
            "state_names_tail": ["goal_dist", "goal_y", "cycle_dist", "cycle_idx"],
        },
        batch_size=4,
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
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_d22idxS_100ep/train_diag_idxS.jsonl").resolve()
        ),
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_g1_omnipicker_clear5_stage2_lora",
        # Stage 2「目标位置」单变量对照（2026-09-22 重启设计）。
        # 与 pi05_g1_omnipicker_clear5_lora **只差一处**：state 多 3 维
        # (dims 18..20 = tgt_rel = 目标 − ee)。数据集同为 181 集、超参逐字相同、
        # 热启动同为 pi05_base，所以两档差异可归因到「模型能不能用上目标位置」。
        #
        # 这 3 维是 **oracle**：target 取每集实测抓取点，等于一个完美感知系统的输出。
        # 工具静止、从第 0 帧就可观测，故不是时间泄漏。结论单边：
        #   oracle 无提升 -> 任何 vision/depth 方案只会更差，整条 depth 路线可砍
        #   oracle 有提升 -> 才值得投入 perception
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # 模型内部恒 32；真实 action 18 维由 padding+裁剪处理
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=G1OmnipickerToolDataConfig(
            repo_id="g1_omnipicker_clear5_stage2_181ep",
            action_dim=18,  # action 未改（用户 Step 3 要求）；state 21 维另由 padding 处理
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,  # 与 orcalab 仿真共存，沿用基线的 4 防 OOM
        num_train_steps=12_000,
        # ⚠ 必须逐字等于 pi05_g1_omnipicker_clear5_lora_s1500（= 基线
        # `clear5_v1_l3090_lora` 实际跑的那条日程）。基线由旧机 train_local_3090.sh
        # 启动，**没有**在命令行覆盖 lr_schedule，所以用的是 _s1500 自身的
        # warmup120/decay1500 —— 即 1500 步后 LR 就贴 2e-5 地板一直到 43500 步
        # （memory 里那句"2e-5 长尾平台"就是这么来的）。
        # 若这里换成 warmup800/decay12000，Stage 2 就多了一个变量，对照作废。
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=120,
            peak_lr=2e-4,
            decay_steps=1_500,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        policy_metadata={
            "state_dim": 21,
            "state_names_tail": ["tgt_rel_x", "tgt_rel_y", "tgt_rel_z"],
        },
        log_interval=10,  # 同基线（只影响打印频率，不影响优化）
        save_interval=500,
        # ⚠ 2026-09-23 改 500 -> 2000：**这是存储参数，不是实验变量**，优化过程一个字节都不受影响。
        #
        # 为什么必须改：checkpoints.py 用 `max_to_keep=1` + `keep_period=K`，语义是
        # "只保最新那个，外加所有 K 的整数倍"（Orbax 日志里的 `Preserving ... (Reason:
        # on keep_period=500)`）。save_interval 也是 500 时，**每个存档都是 500 的整数倍
        # ⇒ 一个都不删**。每个存档实测 8.6G，12000 步 = 24 个 = 206G。
        # 2026-09-22 那晚就是这么崩的：写到 step 8000 的第 16 个时磁盘写满，
        # `ValueError: RESOURCE_EXHAUSTED ... [OS error 28: ENOSPC No space left on device]`，
        # 丢了 5 小时的训练（只救回 500..7500）。
        #
        # 改 2000 后：12000 步只留 2000/4000/6000/8000/10000/12000 + 最新 = 6 个 ≈ 52G。
        # 探针曲线仍拿得到 6 个点，够用。
        keep_period=2_000,
        diag_log_path=str(
            pathlib.Path("./data/g1_omnipicker_clear5_stage2_181ep/train_diag.jsonl").resolve()
        ),
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
