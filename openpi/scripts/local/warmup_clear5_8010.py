#!/usr/bin/env python
"""预热 clear5 serve(8010) 首次推理的 JAX JIT 编译。

只发一条与真实 eval 同结构、像素/state 全零的查询,逼出首次编译,
之后真实首条查询无需再编译、不静置。失败也只是本连接收到 traceback,不影响 serve。
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
CAM_HW = (224, 224, 3)

client = websocket_client_policy.WebsocketClientPolicy(f"ws://{HOST}:{PORT}")
meta = client.get_server_metadata()
print(f"[warmup] connected, server meta keys: {list(meta.keys())}", flush=True)

payload = {
    "images": {
        "cam_head": np.zeros(CAM_HW, dtype=np.uint8),
        "cam_wrist_r": np.zeros(CAM_HW, dtype=np.uint8),
    },
    "state": np.zeros(18, dtype=np.float32),
    "prompt": PROMPT,
}

t0 = time.time()
out = client.infer(payload)
dt = time.time() - t0
acts = np.asarray(out["actions"], dtype=np.float32)
print(f"[warmup] OK 首推耗时 {dt:.1f}s  actions shape={acts.shape} dtype={acts.dtype}", flush=True)
if "server_timing" in out:
    print(f"[warmup] server_timing={out['server_timing']}", flush=True)
print("[warmup] 编译已就绪,真实查询不再需要长编译", flush=True)
