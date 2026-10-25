#!/usr/bin/env python
"""评测图像和训练图像对不上，是"视角不同"还是"约定不同"（翻转/通道序/BGR）？

动机：ik8 把右臂摆到训练帧 0 的**同一个位姿**（实测 0.00mm / 0.00°），图像
**仍然是第 0 百分位**（比 100 个训练集彼此之间都不像，训练集内部 corr ≥ 0.933）。
而且灰度均值在两次评测之间还会跳（ik5 头 81.8 / ik8 头 104.7，训练 88.17~88.71）。
位姿已经排除了，那就要排"是不是根本是同一张画面被转了/换了通道"。

做法：对评测帧做 8 种二面体变换 + 通道置换，各自与训练帧 0 求相关，
看有没有哪一种能把 corr 拉到训练集内部的量级（≥0.9）。
再顺带拿评测帧去和训练集的**每一帧**（ep0 全程 2101 帧，抽稀）比，
看它是否对应到某个**非帧 0**的时刻 —— 若最佳匹配在很后面，说明是时序错位而不是视角错位。
"""
import os

import av
import numpy as np

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
CAM = {"head": "observation.images.cam_head", "wrist": "observation.images.cam_wrist_r"}


def vpath(ep, cam):
    return os.path.join(DS, "videos", "chunk-000", CAM[cam], f"episode_{ep:06d}.mp4")


def frame_at(path, idx):
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i == idx:
                return fr.to_ndarray(format="rgb24")
    return None


def gray(a):
    return a.astype(np.float64).mean(axis=2)


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def variants(a):
    """8 种二面体变换（对 RGB 数组的 H,W 两轴）。"""
    out = {}
    for k in range(4):
        r = np.rot90(a, k, axes=(0, 1))
        out[f"rot{k*90}"] = r
        out[f"rot{k*90}+flipH"] = r[:, ::-1]
    return out


def load_png(path):
    import PIL.Image as I
    return np.asarray(I.open(path).convert("RGB"))


def main():
    ev_dir = "/home/dgl/lxc/.runlogs/task3_frames_ik8"
    for cam, tag in (("head", "head"), ("wrist", "wrist_r")):
        tr0 = frame_at(vpath(0, cam), 0)
        tg = gray(tr0)
        ev = load_png(os.path.join(ev_dir, f"attempt_start_camera_{tag}_color.png"))
        print(f"\n########## cam_{cam} ##########")
        print(f"  训练帧0 灰度 {tg.mean():.2f} | 评测 灰度 {gray(ev).mean():.2f}")
        res = []
        for name, v in variants(ev).items():
            res.append((corr(gray(v), tg), name))
        res.sort(reverse=True)
        for c, name in res:
            print(f"    {name:16s} corr = {c:+.3f}")
        # 通道置换：RGB → BGR / 各通道独立
        for name, v in (("BGR", ev[:, :, ::-1]),
                        ("gray->RGB", np.repeat(gray(ev)[:, :, None], 3, axis=2))):
            print(f"    {name:16s} corr = {corr(gray(v), tg):+.3f}")

        # 评测帧 vs 训练 ep0 全程（抽稀，找最佳匹配时刻）
        print("  评测帧 vs 训练 ep0 全程（每 25 帧取一帧，共约 85 帧）:")
        best = []
        with av.open(vpath(0, cam)) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(st)):
                if i % 25:
                    continue
                best.append((corr(gray(fr.to_ndarray(format="rgb24")), gray(ev)), i))
        best.sort(reverse=True)
        print("    最佳 5 帧: " + ", ".join(f"帧{i}(corr{c:+.3f})" for c, i in best[:5]))


if __name__ == "__main__":
    main()
