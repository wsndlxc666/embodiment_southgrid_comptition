#!/usr/bin/env python
"""模型到底读不读 state 里那三维"周期索引"（dim19 goal_y / dim20 cycle_dist / dim21 cycle_idx）？

用户的问法：index 能不能参与训练、让模型据此知道"该去第几件工具"，从而实现长程记忆。
数据侧已经查清：**能，而且已经在训练里**（d22 的第 19~21 维就是它），且训练里 idx 是按
"右爪张开"这个事件推进的（ep0/1/42 的夹爪跳变帧逐字节相同：115闭 365开 515闭 765开 …
idx 在每次"开"的下一帧 +1）。但"通道存在"不等于"模型在用" —— pi0.5 把 state 拼成
prompt 里一行文字（"State: v0..vN;"），三维混在 19 个浮点里，很可能被当噪声。

本脚本用**训练集自己的 action 当真值**（实测 action[t] == state[t+1]，逐字节相等）来判：
  1) 喂**完全在分布内**的输入（ep0 第 f 帧的图像 + state + ep0 自己的 task 串），
     看预测的 chunk[0] 和录下来的 action[f] 差多少 —— 这是整条评测通路的下限；
  2) 重复同一输入，量出**采样噪声底**（同输入同输出？差多少？）；
  3) **只把 dim19/20/21 换掉**（换成一整轮之前的索引），看输出变化是否超过噪声底。

第 3 步是判据：若"改索引"引起的变化 ≫ 噪声底，说明模型确实读它（用户的想法成立）；
若 ≪ 噪声底，说明模型压根没用这条通道，再改 gate/再训索引都没意义。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_index_channel.py
"""
from __future__ import annotations

import os

import numpy as np

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
CAM = {"cam_head": "observation.images.cam_head",
       "cam_wrist_r": "observation.images.cam_wrist_r"}
EVAL_FRAMES = "/home/dgl/lxc/.runlogs/task3_frames_ik8"
TOOL_REF = np.array([[0.5479, -0.0845, 0.1699], [0.5603, -0.2158, 0.1839],
                     [0.5905, -0.2994, 0.1668], [0.5790, -0.4153, 0.1703],
                     [0.5459, -0.4925, 0.1642]])
TASKS = {0: "把工具全部收进箱子里",
         1: "把桌面上的工具都收进工具箱",
         2: "把所有工具从左到右放回工具箱"}
# ep0 每轮起始帧（= 上一次"右爪张开"的下一帧）
CYCLE_START = [(0, 1), (366, 2), (766, 3), (1166, 4), (1566, 5)]

_STATE_CACHE: dict = {}


def ep_data(ep):
    if ep not in _STATE_CACHE:
        import pandas as pd
        df = pd.read_parquet(f"{DS}/data/chunk-000/episode_{ep:06d}.parquet",
                             columns=["observation.state", "action", "task_index"])
        _STATE_CACHE[ep] = (np.array([list(s) for s in df["observation.state"]], dtype=np.float32),
                            np.array([list(a) for a in df["action"]], dtype=np.float32),
                            int(df["task_index"].iloc[0]))
    return _STATE_CACHE[ep]


def video_frame(ep, cam, idx):
    """取训练视频第 idx 帧，缩到 480x640（与训练录制分辨率一致则原样）。"""
    import av
    p = f"{DS}/videos/chunk-000/{CAM[cam]}/episode_{ep:06d}.mp4"
    with av.open(p) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i == idx:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(idx)


def png_frame(path):
    import PIL.Image as I
    return np.asarray(I.open(path).convert("RGB"))


def to_obs(img_hwc, st, prompt):
    chw = np.transpose(np.ascontiguousarray(img_hwc), (2, 0, 1))
    return {"state": st, "images": {"cam_head": chw, "cam_wrist_r": chw}, "prompt": prompt}


def infer(pol, img, st, prompt):
    a = np.asarray(pol.infer(to_obs(img, st, prompt))["actions"], dtype=np.float32)
    return a.reshape(1, -1) if a.ndim == 1 else a


def main():
    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8010)
    print(f"策略服务器 {pol.get_server_metadata()}")

    S, A, ti = ep_data(0)
    head0 = video_frame(0, "cam_head", 0)
    wrist0 = video_frame(0, "cam_wrist_r", 0)
    pr = TASKS[ti]
    print(f"ep0 的 task 串（逐字用训练那句）= {pr!r}")

    # 两张相机图不同，测试要和真实推理一样两张都给。先各取各的。
    def obs_at(f):
        h = video_frame(0, "cam_head", f)
        w = video_frame(0, "cam_wrist_r", f)
        return (h, w)

    # ---------------- 1) 在分布内：预测 vs 真值 ----------------
    print("\n########## 1) 完全在分布内（ep0 自己的图像+state+task），预测 vs 录制真值 ##########")
    h, w = obs_at(0)
    chw_h = np.transpose(np.ascontiguousarray(h), (2, 0, 1))
    chw_w = np.transpose(np.ascontiguousarray(w), (2, 0, 1))
    obs = {"state": S[0], "images": {"cam_head": chw_h, "cam_wrist_r": chw_w}, "prompt": pr}
    chunk = np.asarray(pol.infer(obs)["actions"], dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None]
    n = min(len(chunk), 50)
    print(f"  chunk 形状 {chunk.shape}；逐帧与录制 action 比（右末端 7:10，单位 mm）")
    errs = []
    for k in range(0, n, 5):
        e = np.linalg.norm(chunk[k, 7:10] - A[k, 7:10]) * 1000
        errs.append(e)
        print(f"    k={k:3d}  预测[{chunk[k,7]:+.4f} {chunk[k,8]:+.4f} {chunk[k,9]:+.4f}]"
              f"  真值[{A[k,7]:+.4f} {A[k,8]:+.4f} {A[k,9]:+.4f}]   误差 {e:7.1f} mm")
    print(f"  前 {n} 帧右末端平均误差 {np.mean(errs):.1f} mm（这就是『整条通路在分布内』的水平）")

    # ---------------- 2) 采样噪声底 ----------------
    print("\n########## 2) 同一输入重复 8 次 → 采样噪声底 ##########")
    outs = []
    for _ in range(8):
        o = np.asarray(pol.infer(obs)["actions"], dtype=np.float32)
        outs.append(o[0] if o.ndim > 1 else o)
    outs = np.array(outs)
    sp = outs[:, 7:10].max(0) - outs[:, 7:10].min(0)
    print(f"  chunk[0] 右末端重复 8 次的极差 = {np.round(sp*1000,1).tolist()} mm"
          f"  (std {np.round(outs[:,7:10].std(0)*1000,1).tolist()} mm)")
    print(f"  ⇒ 噪声底约 {np.linalg.norm(sp)*1000:.1f} mm；比它小的差异都读不出")

    # ---------------- 3) 只换 index 三维 ----------------
    print("\n########## 3) 只换 dim19/20/21（其余逐字节相同）→ 输出变化 ##########")
    print("  第 k 轮用第 k 轮的索引真值，图像/臂位姿**全部冻结在帧 0**：")
    print("  idx   dim19/20/21                 chunk[0] 右末端              "
          "与 idx=1 基准的距离")
    ref = None
    for f, idx in CYCLE_START:
        st = S[0].copy()
        st[18:22] = S[f, 18:22]
        o = np.asarray(pol.infer({"state": st,
                                  "images": {"cam_head": chw_h, "cam_wrist_r": chw_w},
                                  "prompt": pr})["actions"], dtype=np.float32)
        o = o[0] if o.ndim > 1 else o
        if ref is None:
            ref = o[7:10].copy()
        d = np.linalg.norm(o[7:10] - ref) * 1000
        print(f"  {idx:3d}   [{st[19]:+.4f} {st[20]:.4f} {st[21]:.0f}]        "
              f"[{o[7]:+.4f} {o[8]:+.4f} {o[9]:+.4f}]   {d:6.1f} mm")

    # ---------------- 4) 分布外图像 ----------------
    print("\n########## 4) 换成评测 ik8 的头相机图（分布外），同样只换 index ##########")
    try:
        he = png_frame(os.path.join(EVAL_FRAMES, "attempt_start_camera_head_color.png"))
        we = png_frame(os.path.join(EVAL_FRAMES, "attempt_start_camera_wrist_r_color.png"))
        e_h = np.transpose(np.ascontiguousarray(he), (2, 0, 1))
        e_w = np.transpose(np.ascontiguousarray(we), (2, 0, 1))
        ref2 = None
        for f, idx in CYCLE_START:
            st = S[0].copy()
            st[18:22] = S[f, 18:22]
            o = np.asarray(pol.infer({"state": st, "images": {"cam_head": e_h,
                                                              "cam_wrist_r": e_w},
                                      "prompt": pr})["actions"], dtype=np.float32)
            o = o[0] if o.ndim > 1 else o
            if ref2 is None:
                ref2 = o[7:10].copy()
            print(f"  idx={idx}   [{o[7]:+.4f} {o[8]:+.4f} {o[9]:+.4f}]   "
                  f"与 idx=1 的距离 {np.linalg.norm(o[7:10]-ref2)*1000:6.1f} mm"
                  f"   最近的 TOOL_REF #{int(np.argmin([np.linalg.norm(o[7:10]-p) for p in TOOL_REF]))+1}")
    except Exception as exc:
        print(f"  跳过（{type(exc).__name__}: {exc}）")


if __name__ == "__main__":
    main()
