#!/usr/bin/env python
"""评测帧 0 的图像，到底落在**训练集自己的帧 0 分布**里面还是外面？

为什么需要这个：之前只量了「评测帧0 vs 训练ep0帧0」的相关系数（cam_head +0.058、
cam_wrist_r −0.061），但**一个样本和另一个样本不像，本身不说明任何问题** —— 训练集
100 集自己的帧 0 之间也可能不像（工具摆位、起始姿态若有随机化）。决定性的问法是：

    拿 100 个**训练**集的帧 0 两两比，得到一个"像"的分布；
    再看评测那一帧落在这个分布的第几百分位。

落在外面的 ⇒ 图像确实是分布外的，"视觉没起作用"有了物证；
落在里面 ⇒ 图像没问题，问题在别处（prompt/state/动作层）。

同时把 100 集的 state[7:10]（右末端，base 系）帧 0 也摊开，看起始位姿的散布 ——
用来判断"对齐到 3.16mm 还是 9.68mm"这点差别在训练集里算不算事。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_image_indist.py \
      --eval_frames /home/dgl/lxc/.runlogs/task3_frames_ik5 --n 100
"""
import argparse
import io
import os

import av
import numpy as np

DS = "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep"
CAM = {"head": "observation.images.cam_head", "wrist": "observation.images.cam_wrist_r"}


def video_path(ep, cam, chunk=0):
    return os.path.join(DS, "videos", f"chunk-{chunk:03d}", CAM[cam],
                        f"episode_{ep:06d}.mp4")


def frame_at(path, idx):
    """取第 idx 帧，RGB uint8 HxWx3。"""
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
    x, y = a.ravel(), b.ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def load_png_gray(path):
    import PIL.Image as I
    return gray(np.asarray(I.open(path).convert("RGB")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_frames", default="/home/dgl/lxc/.runlogs/task3_frames_ik5")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--idx", type=int, default=0, help="比第几帧（0=开局）")
    args = ap.parse_args()

    import pandas as pd

    print(f"参照帧 = 第 {args.idx} 帧；训练集 {DS}")
    for cam in ("head", "wrist"):
        ref = frame_at(video_path(0, cam), args.idx)
        if ref is None:
            print(f"[{cam}] 取不到训练 ep0 的帧 {args.idx}，跳过")
            continue
        rg = gray(ref)
        cs, brights = [], []
        for ep in range(args.n):
            f = frame_at(video_path(ep, cam), args.idx)
            if f is None:
                continue
            g = gray(f)
            cs.append(corr(g, rg))
            brights.append(float(g.mean()))
        cs = np.asarray(cs)
        brights = np.asarray(brights)
        print(f"\n===== cam_{cam}（{len(cs)} 集）=====")
        print(f"  训练集内部：corr(ep_i 帧{args.idx}, ep0 帧{args.idx}) "
              f"min={cs.min():+.3f}  p5={np.percentile(cs,5):+.3f}  "
              f"中位={np.median(cs):+.3f}  p95={np.percentile(cs,95):+.3f}  max={cs.max():+.3f}")
        print(f"  训练集灰度均值：{brights.min():.2f}~{brights.max():.2f}（中位 {np.median(brights):.2f}）")

        ev = os.path.join(args.eval_frames, f"attempt_start_camera_{'head' if cam=='head' else 'wrist_r'}_color.png")
        if os.path.exists(ev):
            eg = load_png_gray(ev)
            c = corr(eg, rg)
            pct = float((cs < c).mean() * 100)
            print(f"  评测 attempt_start：corr={c:+.3f}  灰度均值={eg.mean():.2f}"
                  f"  → 落在训练集第 {pct:.1f} 百分位"
                  f"（'低于它的训练集占比'）")
            print(f"  评测 vs 各训练集：比 {int((cs<c).sum())}/{len(cs)} 集都不像")
        else:
            print(f"  （没找到 {ev}）")

    # 起始位姿散布
    print("\n===== 100 集帧 0 的右末端 state[7:10]（base 系）=====")
    rows = []
    for ep in range(args.n):
        p = os.path.join(DS, "data", "chunk-000", f"episode_{ep:06d}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["observation.state", "frame_index"])
        df = df[df["frame_index"] == args.idx]
        if len(df) == 0:
            continue
        rows.append(np.asarray(df["observation.state"].iloc[0], dtype=np.float64))
    if rows:
        S = np.asarray(rows)
        rp = S[:, 7:10]
        print(f"  右末端帧{args.idx}：x {rp[:,0].min():.4f}~{rp[:,0].max():.4f}"
              f"  y {rp[:,1].min():.4f}~{rp[:,1].max():.4f}"
              f"  z {rp[:,2].min():.4f}~{rp[:,2].max():.4f}")
        c0 = rp.mean(axis=0)
        d = np.linalg.norm(rp - c0, axis=1)
        print(f"  到均值的距离：中位 {np.median(d)*1000:.1f}mm  p95 {np.percentile(d,95)*1000:.1f}mm"
              f"  max {d.max()*1000:.1f}mm")
        # 逐维标准差（mm）
        print(f"  逐维 std (mm)：{np.round(rp.std(axis=0)*1000, 2).tolist()}")
        print(f"  帧{args.idx} 的姿态 r_quat std：{np.round(S[:,10:14].std(axis=0), 5).tolist()}")


if __name__ == "__main__":
    main()
