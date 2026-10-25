"""把 d22 训练集的一集逐帧喂回 G1OmniPickerLeRobotStorage.build_state，
比对现算出来的 22 维和数据集里存的是否逐位一致。

为什么必须有这一步：state 是**有状态**的（cycle_idx 靠数开爪事件推进）。训练端
是在离线第二遍扫描里按"整集的 5 个开爪边沿"切轮次，推理端只能一帧一帧在线数。
两者差一帧、或者边沿判据不同，模型收到的就是它从没见过的输入 —— 而且**不报错**。
d19 那次就是栽在训练/推理维度不一致上，且是静默的。

跑法（cwd 必须是 SouthGrid/src）：
    python scripts/verify_d22_storage_replay.py [集号...]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage  # noqa: E402

DATA = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
REFS = json.load(open(os.path.join(DATA, "tool_refs.json"), encoding="utf-8"))


def main(eps):
    from conf import g1_omnipicker_conf as gc

    n_l = len(gc.gripper_l["actuator_names"])
    n_r = len(gc.gripper_r["actuator_names"])
    l_rng = np.array([r[:2] for r in gc.gripper_l["actuator_ranges"][:n_l]], dtype=np.float64)
    r_rng = np.array([r[:2] for r in gc.gripper_r["actuator_ranges"][:n_r]], dtype=np.float64)

    tmp = tempfile.mkdtemp(prefix="_verify_d22_")
    st = G1OmniPickerLeRobotStorage(
        dataset_path=tmp,
        tool_refs=np.asarray(REFS["goals"], dtype=np.float32),
        ref_name="goal_dist",
        cycle_refs=np.asarray(REFS["tool_refs"], dtype=np.float32),
    )
    print(f"state_dim={st.state_dim}  names={st.state_names[18:]}")
    assert st.state_dim == 22, f"state_dim 应为 22，实际 {st.state_dim}"

    all_ok = True
    for ep in eps:
        df = pd.read_parquet(os.path.join(DATA, "data", "chunk-000", f"episode_{ep:06d}.parquet"))
        s = np.asarray(df["observation.state"].tolist(), dtype=np.float64)   # (T, 22)
        st.reset_cycle()

        got = np.empty_like(s)
        for t in range(s.shape[0]):
            pos = np.stack([s[t, 0:3], s[t, 7:10]]).astype(np.float32)          # (2,3)
            quat = np.stack([s[t, 3:7], s[t, 10:14]]).astype(np.float32)        # (2,4)
            # state[14:18] 存的就是归一化后的 [0,1]，反解回电机原始值再交给
            # build_state —— 它内部会再归一化一遍，这条往返路径必须无损。
            ln, rn = s[t, 14:16], s[t, 16:18]
            motor = np.concatenate([
                ln * (l_rng[:, 1] - l_rng[:, 0]) + l_rng[:, 0],
                rn * (r_rng[:, 1] - r_rng[:, 0]) + r_rng[:, 0],
            ]).astype(np.float32)
            got[t] = st.build_state({
                "/action/end/position": pos,
                "/action/end/orientation": quat,
                "/action/effector/motor": motor,
            })

        # 前 18 维经过 float32→float64 往返，只比后 4 维（18..21）的**语义**部分。
        d18 = np.abs(got[:, 18] - s[:, 18]).max()
        rest = np.abs(got[:, 19:22] - s[:, 19:22])
        per = rest.max(axis=0)
        idx_eq = int((got[:, 21] == s[:, 21]).sum())
        ok = per.max() < 1e-4 and d18 < 1e-5 and idx_eq == len(s)
        all_ok &= ok
        print(f"\nep{ep:03d}  T={len(s)}  {'OK ' if ok else 'MISMATCH'}")
        print(f"  dim18 goal_dist  最大误差 {d18:.3e}")
        print(f"  dim19 goal_y     最大误差 {per[0]:.3e}")
        print(f"  dim20 cycle_dist 最大误差 {per[1]:.3e}")
        print(f"  dim21 cycle_idx  逐帧相等的帧数 {idx_eq}/{len(s)}"
              f"  推理端取值 {sorted(set(got[:, 21].astype(int)))}"
              f"  训练端取值 {sorted(set(s[:, 21].astype(int)))}")

    print("\n" + ("全部一致：推理端与训练端逐帧对齐。" if all_ok else
                  "**有偏差**：推理端现算的 3 维与训练数据不符，不能拿去评 d22 模型。"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main([int(x) for x in sys.argv[1:]] or [0, 37, 99]))
