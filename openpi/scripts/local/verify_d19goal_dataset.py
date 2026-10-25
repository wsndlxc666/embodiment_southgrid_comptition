#!/usr/bin/env python
"""端到端验证 d19-goal 数据集：走 openpi 自己的 data loader 拉一个 batch。

为什么必须写成文件而不是 `python - <<EOF`：
    LeRobot 的视频解码走 multiprocessing 的 **spawn** 启动方式，子进程会重新
    import `__main__`。stdin 喂进来的脚本 `__main__.__file__` 是 `<stdin>`，
    子进程去 open 这个路径直接 FileNotFoundError，看起来像数据集坏了，其实是
    测试方式的问题。train.py 是真文件所以训练不受影响。

检查项：
  1. config 能被解析, action_dim(数据)=19 / action_dim(模型)=32
  2. norm_stats.json 落在 loader 实际去找的位置
  3. 能真的取出一个 batch, state/action 是 19 维
  4. 第 19 维归一化后落在 [-1, 1], 且和手工按公式算的值一致

用法:  uv run python scripts/local/verify_d19goal_dataset.py [config_name]
"""
import json
import pathlib
import sys

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _dl


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "pi05_g1_omnipicker_clear5_d19goal_lora_20k"
    cfg = _config.get_config(name)
    print(f"[1] config 解析成功: {cfg.name}")
    print(f"    repo_id={cfg.data.repo_id}  action_dim(数据)={cfg.data.action_dim}  "
          f"action_dim(模型)={cfg.model.action_dim}")

    p = pathlib.Path(cfg.assets_dirs) / cfg.data.repo_id / "norm_stats.json"
    print(f"[2] norm_stats: {'存在' if p.exists() else '>>> 缺失 <<<'}  {p}")
    if not p.exists():
        raise SystemExit("norm_stats 不在 loader 会找的位置, 训练起不来")
    ns = json.loads(p.read_text())["norm_stats"]

    dl = _dl.create_data_loader(cfg, sharding=None, shuffle=False)
    obs, act = next(iter(dl))
    state = np.asarray(obs.state)
    action = np.asarray(act)
    n_data = cfg.data.action_dim          # 数据集真实维数(19)
    n_model = cfg.model.action_dim        # 模型输入维数(32)
    print(f"[3] 取出一个 batch: state {state.shape}  action {action.shape}")
    print(f"    注意最后一维是 {state.shape[-1]} 不是 {n_data} —— 这是对的。"
          f"transform 顺序是")
    print(f"    Normalize -> TokenizePrompt(此时还是 {n_data} 维) -> PadStatesAndActions"
          f"(补到 {n_model})")
    print(f"    images: {[(k, np.asarray(v).shape) for k, v in obs.images.items()]}")
    print(f"    tokenized_prompt: {tuple(np.asarray(obs.tokenized_prompt).shape)}")

    ok = True
    # --- 补零维必须恰好是 0.0（pad 发生在 Normalize 之后, 所以是字面 0, 不再被归一化） ---
    pad = state[..., n_data:]
    if not np.all(pad == 0.0):
        print(f"    >>> 补零维 {n_data}..{n_model - 1} 不全是 0: "
              f"min={pad.min():.3e} max={pad.max():.3e} <<<")
        ok = False
    else:
        print(f"[4] 补零维 {n_data}..{n_model - 1} 全为 0.0 ✅")

    # --- 第 19 维必须是 goal_dist: 反归一化后应与 parquet 里的 d 逐值吻合 ---
    # 这是本次改动的核心断言。不能只看"落在合理范围"——那证明不了这一维没被别的列串位。
    q01 = np.asarray(ns["state"]["q01"])[:n_data]
    q99 = np.asarray(ns["state"]["q99"])[:n_data]
    raw = (state[..., :n_data] + 1.0) * (q99 - q01) / 2.0 + q01

    import glob
    import pyarrow.parquet as pq
    pool = []
    for f in sorted(glob.glob(f"/home/dgl/lxc/openpi/data/{cfg.data.repo_id}/data/chunk-*/episode_*.parquet"))[:8]:
        pool.append(np.array(pq.read_table(f, columns=["observation.state"])["observation.state"].to_pylist(),
                             dtype=np.float64)[:, 18])
    pool = np.concatenate(pool)                       # 前 8 集全部的 d
    got = raw[..., 18].ravel()
    # 每个取出的值都应能在真实 d 集合里找到几乎相同的值(1e-5 内)
    err = np.array([np.abs(pool - v).min() for v in got])
    print(f"[5] 第19维反归一化后: [{got.min():.4f}, {got.max():.4f}]  "
          f"(数据集 d 真实范围 [{pool.min():.4f}, {pool.max():.4f}])")
    print(f"    每个值到真实 d 集合的最近距离: 最大 {err.max():.2e}  "
          f"{'✅ 就是 goal_dist' if err.max() < 1e-4 else '❌ 对不上, 这维不是 goal_dist'}")
    if err.max() >= 1e-4:
        ok = False
    # 另外反归一化整个 19 维, 和前 8 集逐帧比, 确认没有串位。
    # 比对池要覆盖多集: 批次未必来自第 0 集(shuffle=False 也不保证从 ep0 开头取)。
    eps = []
    for f in sorted(glob.glob(f"/home/dgl/lxc/openpi/data/{cfg.data.repo_id}/data/chunk-*/episode_*.parquet"))[:8]:
        eps.append(np.array(pq.read_table(f, columns=["observation.state"])["observation.state"].to_pylist(),
                            dtype=np.float64))
    ep_pool = np.concatenate(eps, axis=0)
    vecs = raw[..., :n_data].reshape(-1, n_data)
    hits = [np.abs(ep_pool - v[None, :]).max(axis=1).min() < 1e-4 for v in vecs]
    hit = sum(hits)
    print(f"    整个 {n_data} 维向量能在前8集里精确匹配: {hit}/{len(got)} "
          f"{'✅ 无串位' if hit == len(got) else '❌ 有样本对不上任何一帧'}")
    if hit != len(got):
        ok = False

    print("\n结论:", "✅ 数据集可用, 可以起训" if ok else "❌ 有问题, 先别起训")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
