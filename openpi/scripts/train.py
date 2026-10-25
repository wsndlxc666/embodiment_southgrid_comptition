import dataclasses
import functools
import json
import logging
import platform
import subprocess
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def log_gpu_state(tag: str):
    """把当前 GPU 状态(温度/功耗/显存/频率)写进日志，用于定位 GPU 故障。

    在崩溃点调用时, 会给出"崩溃瞬间 GPU 还活着没有"的第一手证据:
    - 若 nvidia-smi 仍能返回温度/功耗 → 只是某个 kernel 出错, GPU 本身没死
    - 若 nvidia-smi 报 "No devices were found / Unknown Error" → GPU 被驱动锁死,
      属于硬件/驱动级问题(过热、供电、显存坏、驱动 bug), 需要重启甚至查硬件
    """
    logging.info(f"[GPU:{tag}] 采集 GPU 现场状态 ...")
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,"
                "utilization.gpu,memory.used,memory.total,clocks.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                logging.info(f"[GPU:{tag}] {line.strip()}")
        else:
            logging.error(f"[GPU:{tag}] nvidia-smi 返回码={out.returncode}")
            logging.error(f"[GPU:{tag}] stderr={out.stderr.strip()}")
            logging.error(
                f"[GPU:{tag}] nvidia-smi 无法访问 GPU → 典型驱动/硬件锁死症状, 建议: 重启 + sudo dmesg | grep -i nvidia"
            )
    except Exception as e:  # noqa: BLE001
        logging.error(f"[GPU:{tag}] nvidia-smi 调用失败: {e}")


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # 逐维误差先算出来：加权求和得到用于反传的 loss，未加权的逐维均值作为 aux
        # 返回供监控(看爪维/距离维到底有没有在学)。
        per_dim = model.compute_loss_per_dim(rng, observation, actions, train=True)
        # 加权那一项走模型自己的单一入口(逐维权重 × 合爪帧权重)，别在这里重写一遍 ——
        # 两处各写一份的话，改了 loss 公式却漏了训练路径，读数会看着没变。
        # 它只做加权、不做前向，所以复用上面这份 per_dim，不会多跑一次前向。
        if hasattr(model, "weight_loss_per_dim"):
            weighted = model.weight_loss_per_dim(per_dim, actions)
        elif hasattr(model, "apply_loss_weights"):
            weighted = model.apply_loss_weights(per_dim)
        else:
            weighted = per_dim
        return jnp.mean(weighted), jnp.mean(per_dim, axis=(0, 1))

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 注意 flax.nnx 的返回约定和裸 jax 不一样：nnx 是 ((value, aux), grads)，
    # 裸 jax.value_and_grad 才是 (value, (grads, aux))。nnx 这边 grads 是 State。
    (loss, loss_per_dim), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "loss_per_dim": loss_per_dim,
    }
    # 合爪帧加权开着时，把"这一批有多少帧吃到了它"打出来。不加这一行的话，这一项生效与否
    # 在日志里完全看不出来 —— 逐维那几列是**未加权**的，权重再大它也不动。
    _m = getattr(config, "model", None)
    if getattr(_m, "grasp_frame_weight", 1.0) > 1.0:
        info["grasp_frac"] = jnp.mean(
            (actions[..., _m.grasp_grip_dim] > _m.grasp_close_thresh).astype(jnp.float32)
        )
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    log_gpu_state("start")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    logging.info("[Phase] 初始化数据加载器 ...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    logging.info("[Phase] 训练状态就绪，进入训练循环(JIT 编译约需 20s，随后每步约 3.8s) ...")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # 逐维 loss 单独收集：它是向量，混进 infos 会被 stack_forest 的标量平均压扁。
    dim_infos = []
    diag_path = getattr(config, "diag_log_path", None)
    for step in pbar:
        with sharding.set_mesh(mesh):
            try:
                train_state, info = ptrain_step(train_rng, train_state, batch)
            except Exception:
                logging.error(f"[CRASH] 训练在 step={step} 崩溃(ptrain_step GPU 内核执行阶段)! 采集现场信息:")
                log_gpu_state(f"crash@step{step}")
                raise
        dim_infos.append(jax.device_get(info.pop("loss_per_dim")))
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            # 关键几维单独打出来，方便在进度条旁直接盯
            dim_mean = np.stack(dim_infos).mean(axis=0)
            info_str += ("  | 逐维: 右爪16={:.5f} 右爪17={:.5f} 距离18={:.5f} 左爪14={:.5f} 右臂7={:.5f}").format(
                dim_mean[16], dim_mean[17], dim_mean[18], dim_mean[14], dim_mean[7]
            )
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if diag_path:
                record = {"step": int(step), "loss": float(reduced_info["loss"]),
                          "grad_norm": float(reduced_info["grad_norm"])}
                record.update({f"dim{i:02d}": float(v) for i, v in enumerate(dim_mean)})
                with open(diag_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
            infos = []
            dim_infos = []
        try:
            batch = next(data_iter)
        except Exception:
            logging.error(f"[CRASH] 数据加载在 step={step} 附近崩溃(next(data_iter) CPU/设备搬运阶段)! 采集现场信息:")
            log_gpu_state(f"data@step{step}")
            raise

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
