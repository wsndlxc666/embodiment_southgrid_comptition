"""开局视角验收：④ 的开局画面和训练帧 0 差多少？—— 一个**和得分无关**的定量判据。

为什么需要它（2026-09-17）：
  `--start_align pos` 时右末端姿态差 **141.59°**。腕相机 `cam_wrist_r` 就装在腕上，
  姿态差 140° 等于把它的视向整个转掉。实测两个相机的开局画面 vs 训练帧 0：

      cam_head     相关 +0.787   ← 同分布（所以模型看得见桌面，能走出大概的轨迹形状）
      cam_wrist_r  相关 +0.018   ← **完全无关**，两张不相干的图

  这解释了用户问的「知道位置、也有距离，怎么就过不去」和「视觉根本没起作用」：
  腕视角落在训练分布之外，而抓取/释放恰恰依赖腕视角。头视角没坏，所以症状是
  「轨迹形状对、抓错工具、抓着不放」而不是「完全乱走」。

  验收标准：姿态对齐做好之后，cam_wrist_r 的相关度应当**大幅上升**（向 cam_head 看齐）。
  得分受太多因素影响、噪声大；这个指标直接量"起始观测是不是回到分布内"，快且稳。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/lxc/openpi/.venv/bin/python \
      src/examples/inference/g1_omnipicker/probe_start_view.py \
      --frames-dir /home/dgl/.runlogs/task3_frames_ik
"""
from __future__ import annotations

import argparse
import os

import numpy as np

CAM_FILES = {
    "cam_head": "attempt_start_camera_head_color.png",
    "cam_wrist_r": "attempt_start_camera_wrist_r_color.png",
}


def train_frame(dataset: str, key: str, episode: int, frame: int) -> np.ndarray:
    """用 av 解训练视频的某一帧（RGB, HWC）。

    不用 lerobot 的 decode_video_frames：本机上它对这批 HEVC 静默返回全零。
    """
    import av
    path = os.path.join(dataset, "videos/chunk-000",
                        f"observation.images.{key}", f"episode_{episode:06d}.mp4")
    c = av.open(path)
    out = None
    for i, fr in enumerate(c.decode(video=0)):
        if i == frame:
            out = fr.to_ndarray(format="rgb24")
            break
    c.close()
    if out is None:
        raise RuntimeError(f"{path} 没有第 {frame} 帧")
    if out.mean() < 1.0:
        raise RuntimeError(f"{path} 第 {frame} 帧是全黑，解码器有问题")
    return out


def sim_frame(frames_dir: str, key: str) -> np.ndarray:
    """④ 存盘时做过 RGB2BGR，这里逆回来。"""
    import cv2
    bgr = cv2.imread(os.path.join(frames_dir, CAM_FILES[key]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"{frames_dir}/{CAM_FILES[key]} 读不到")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compare(a: np.ndarray, b: np.ndarray) -> tuple:
    import cv2
    g = lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32),
                             (32, 24))
    ga, gb = g(a), g(b)
    d = np.abs(ga - gb)
    corr = float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])
    return corr, float(d.mean()), float(a.mean()), float(b.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--frames-dir", default="/home/dgl/lxc/.runlogs/task3_frames")
    ap.add_argument("--label", default="", help="这一档的名字，只影响打印")
    ap.add_argument("--show", action="store_true", help="额外打印 32x24 亮度字符图")
    args = ap.parse_args()

    tag = args.label or os.path.basename(args.frames_dir.rstrip("/"))
    print(f"=== 开局视角验收：{tag} ===")
    print(f"  训练参考 = {args.dataset} ep{args.episode} 帧{args.frame}")
    print(f"  仿真画面 = {args.frames_dir}")
    print(f"\n{'相机':<12}{'相关':>8}{'灰度差':>9}{'训练亮度':>10}{'仿真亮度':>10}   判定")
    worst = None
    for key in CAM_FILES:
        try:
            a = train_frame(args.dataset, key, args.episode, args.frame)
            b = sim_frame(args.frames_dir, key)
        except Exception as e:
            print(f"{key:<12}  读图失败: {e}")
            continue
        corr, gd, ma, mb = compare(a, b)
        worst = corr if worst is None else min(worst, corr)
        if corr > 0.5:
            v = "同分布 ✓"
        elif corr > 0.25:
            v = "边界 △"
        else:
            v = "**不在分布内 ✗**"
        print(f"{key:<12}{corr:>+8.3f}{gd:>9.2f}{ma:>10.2f}{mb:>10.2f}   {v}")
        if args.show:
            ch = " .:-=+*#%@"
            gg = lambda x: __import__("cv2").resize(
                __import__("cv2").cvtColor(x, __import__("cv2").COLOR_RGB2GRAY)
                .astype(np.float32), (32, 24))
            ga, gb = gg(a), gg(b)
            print(f"    {'训练':<34}|  仿真")
            for r in range(0, 24, 3):
                print("    " + "".join(ch[min(9, int(v / 25.6))] for v in ga[r])
                      + "  |  " + "".join(ch[min(9, int(v / 25.6))] for v in gb[r]))
    if worst is None:
        print("\n  ⚠ 一张图都没读到，无法判定（检查 --frames-dir 是不是写错了）")
    else:
        print(f"\n  最差相关 = {worst:+.3f}   "
              f"（腕相机是抓取/释放的关键视角；它低就是起始姿态没对上）")


if __name__ == "__main__":
    main()
