#!/usr/bin/env python
"""预热 + 接口验证：d19(19 维 state) 模型的 serve(8010)。

和老的 warmup_clear5_8010.py 的区别：state 发 19 维（第 19 维 = 右臂末端到最近
工具的距离），并顺带发一次 18 维做对照 —— 老仿真如果还没改，就是 18 维，
这里能直接看出服务端会怎么处理。

用法: warmup_d19_8010.py
"""
import sys
import time

import numpy as np

_openpi_client_src = "/home/dgl/lxc/openpi/packages/openpi-client/src"
if _openpi_client_src not in sys.path:
    sys.path.insert(0, _openpi_client_src)

from openpi_client import websocket_client_policy  # noqa: E402

HOST, PORT = "localhost", 8010
PROMPT = "把桌面上的工具都收进工具箱"
# 必须和真实推理的相机分辨率一致(dataStorage/lerobot_camera.py: DEFAULT_HW=(480,640))。
# 曾写成 (224,224,3)：JIT 是按 shape 缓存的，预热用一个 shape、真跑用另一个，
# 服务端会在首个真实请求上**重新编译一次**，预热等于白做，还多担一次
# keepalive 1011 掉线(编译 ~30s 占住主线程)。
CAM_HW = (480, 640, 3)


def make_state(dim: int) -> np.ndarray:
    """造一条"看起来合理"的 state：归一化空间里 0 就是中位值。"""
    return np.zeros(dim, dtype=np.float32)


def main() -> int:
    client = websocket_client_policy.WebsocketClientPolicy(f"ws://{HOST}:{PORT}")
    meta = client.get_server_metadata()
    print(f"[warmup] 已连接, server meta keys: {list(meta.keys())}", flush=True)

    rc = 0
    for dim, tag in ((19, "正确接口"), (18, "老接口(对照)")):
        payload = {
            "images": {
                "cam_head": np.zeros(CAM_HW, dtype=np.uint8),
                "cam_wrist_r": np.zeros(CAM_HW, dtype=np.uint8),
            },
            "state": make_state(dim),
            "prompt": PROMPT,
        }
        t0 = time.time()
        try:
            out = client.infer(payload)
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[warmup] state={dim:2d} 维 ({tag}) -> 服务端报错 ({dt:.1f}s): {type(e).__name__}: {str(e)[:300]}", flush=True)
            rc = 1
            continue
        dt = time.time() - t0
        acts = np.asarray(out["actions"], dtype=np.float32)
        print(f"[warmup] state={dim:2d} 维 ({tag}) -> OK {dt:5.1f}s  actions shape={acts.shape} dtype={acts.dtype}", flush=True)
        if dim == 19:
            print(f"         第 1 帧 action 形状={acts[0].shape}", flush=True)
            print(f"         右爪16={acts[0][16]:+.4f} 右爪17={acts[0][17]:+.4f} 距离18={acts[0][18]:+.4f}", flush=True)
            print(f"         动作范围 [{acts.min():+.3f}, {acts.max():+.3f}]  均值 {acts.mean():+.4f}", flush=True)
        if "server_timing" in out:
            print(f"         server_timing={out['server_timing']}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
