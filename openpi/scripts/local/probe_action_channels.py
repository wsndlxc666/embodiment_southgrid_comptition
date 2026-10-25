#!/usr/bin/env python
"""验证:闭爪指令有没有活着走到模型的训练目标(actions)里。

走训练完全相同的变换链(create_torch_dataset -> transform_dataset),
但 num_workers=0 单进程跑,避免 spawn 又要 __main__ 保护。
逐样本打印 actions 张量的爪子通道,确认变换链没有把它丢掉或压平。
"""
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.training import config as _config
import openpi.training.data_loader as _dl

CONFIG = "pi05_g1_omnipicker_clear5_lora_s1500"
N_BATCH = 45


def main():
    cfg = _config.get_config(CONFIG)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f"[probe] repo_id = {data_config.repo_id}", flush=True)

    # ---- 关键:把视频解码换掉。图片与"爪子动作通不通"无关,省掉 HEVC 逐帧 seek ----
    _orig_query_videos = lerobot_dataset.LeRobotDataset._query_videos

    def _stub_query_videos(self, query_indices, ep_idx):
        out = _orig_query_videos(self, query_indices, ep_idx)
        for k in out:
            out[k] = np.zeros_like(out[k])
        return out

    lerobot_dataset.LeRobotDataset._query_videos = _stub_query_videos

    loader = _dl.create_torch_data_loader(
        data_config,
        model_config=cfg.model,
        action_horizon=cfg.model.action_horizon,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        framework="pytorch",
    )
    print("[probe] loader 建好", flush=True)

    it = iter(loader)
    # 归一化参考: 开=0.048 → -1.0 ; 闭=1.000 → +1.0
    print(f"{'批':>3} {'样':>3} | {'右爪[0]':>8} {'右爪[14]':>8} {'右爪[17]':>8} | "
          f"{'左爪[0]':>8} | {'右臂[0,7]':>8} | {'填充18':>7} {'填充31':>7} | 状态右爪")
    print("-" * 104)

    seen_open = seen_close = 0
    gmin, gmax = 9e9, -9e9
    A = None
    for bi in range(N_BATCH):
        obs, act = next(it)
        A = np.asarray(act)
        S = np.asarray(obs.state)
        for si in range(A.shape[0]):
            g16 = A[si, 0, 16]      # 右爪 内指
            g17 = A[si, 0, 17]      # 右爪 外指
            g14 = A[si, 0, 14]      # 左爪 内指
            arm = A[si, 0, 7]       # 右臂 x
            pad18, pad31 = A[si, 0, 18], A[si, 0, 31]
            st_g = S[si, 16]
            if g16 > 0.5:
                seen_close += 1
            else:
                seen_open += 1
            gmin, gmax = min(gmin, g16), max(gmax, g16)
            print(f"{bi:>3} {si:>3} | {g16:>8.4f} {g14:>8.4f} {g17:>8.4f} | "
                  f"{g14:>8.4f} | {arm:>8.4f} | {pad18:>7.3f} {pad31:>7.3f} | {st_g:>8.4f}",
                  flush=True)
    print("-" * 104)
    print(f"actions 形状 = {A.shape}  (batch, horizon, dim)")
    print(f"共 {seen_open + seen_close} 个样本: 右爪归一化 ≤0.5 的 {seen_open} 个, >0.5 的 {seen_close} 个")
    print(f"右爪归一化取值范围 = [{gmin:+.4f}, {gmax:+.4f}]")
    print("参考: 开=0.048 → 归一化 -1.00 ; 闭=1.000 → 归一化 +1.00")


if __name__ == "__main__":
    main()
