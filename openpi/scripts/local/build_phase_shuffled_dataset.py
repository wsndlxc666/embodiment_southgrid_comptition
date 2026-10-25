#!/usr/bin/env python3
"""造"相位重排"数据集 d22idxS：把每个 episode 的 4 个工具段重新排序，打破"相位=节拍器"。

## 为什么要造这个（2026-09-20）

实测原数据集：**100/100 集的 cycle_idx 跳变帧完全相同**（366/766/1166/1566/1966），
每集都是 2101 帧。也就是说 `cycle_idx` 在训练里是**帧号的纯函数** —— 它携带的信息
不超过"图像进度"本身。模型于是学成"按视觉进度走一张时间表"，而不是"读相位决定去
拿哪件"。部署时 harness 一旦按别的规则推进相位（实测：什么都没入箱就推了两次），
图像与 idx 互相矛盾，模型收到训练里从没出现过的组合 → 死循环。

这正对上 memory `openpi-index-channel-is-weak` §4 早就写下的修法：
**"随机化工具顺序，让同一画面出现在不同 index 下"**。

## 做法（为什么这么小）

原轨迹结构（cycle_idx 实测）：
    [seg0: home→扳手→箱][seg1][seg2][seg3][seg4][seg5: 箱→home]
      idx=1              2     3     4     5      6

- `seg0` 从 **home 位姿**出发，**必须排第一**（拼到别处会造 630~680mm 跳变）
- `seg1..seg4` 都是"箱→工具→箱"，**两端都在箱位**，可以任意互换
- `seg5` 是收尾，排最后

所以重排 = `[seg0] + perm(seg1..seg4) + [seg5]`，24 种排列轮着用。

**只改 dim21（cycle_idx）这一维**，因为：
- dim18 `goal_dist`、dim20 `cycle_dist` 只依赖末端位置与目标工具，重排不改目标
- dim19 `goal_y` = **被拿那件工具的 y**，重排后仍然是那件工具 ⇒ 不变
- 只有 dim21 要改成"新的第几位"

⇒ 训练里 `(idx, goal_y)` 仍是**自洽的一对**，但 idx 不再能通过固定表推出 goal_y。
模型想拿对工具，就**只能读 goal_y**（部署时 runner 按 `cycle_refs[idx-1]` 算，永远自洽）。
这才是这一轮的作用：**把信息从图像手里抢到相位通道手里**。

## 拼缝大小（`measure_splice_gap.py` 实测）

`seg1..seg4` 互换的拼缝 0~105mm，裁帧也压不下去；而**原数据单帧最大位移本来就是
111mm**（最大速度 2225mm/s）⇒ 拼缝落在数据自身运动分布内，**不做平滑**。

## action 怎么处理

原数据 `action[t] == state[t+1]` **逐字节成立**（已复核 max|diff|=0.0）。
重排后直接令 `new_action[t] = new_state[t+1]`（末行沿用原 action[-1]），
这样这条恒等式在新顺序下**继续逐字节成立**，且 dim19/20/21 自动跟着走对。

用法:
    build_phase_shuffled_dataset.py --limit 2      # 先造 2 集验证
    build_phase_shuffled_dataset.py                # 全量 100 集
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
DST = Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep")
BOUND = [0, 366, 766, 1166, 1566, 1966, 2101]     # 实测：100/100 集相同
FPS = 20
CAMS = ["observation.images.cam_head", "observation.images.cam_wrist_r"]

# 24 种排列，作用于 seg1..seg4（seg0 恒第一、seg5 恒最后）
PERMS = list(itertools.permutations(range(1, 5)))
assert len(PERMS) == 24


def perm_for(ep: int) -> tuple[int, ...]:
    return PERMS[ep % len(PERMS)]


def frame_order(ep: int) -> list[int]:
    """新顺序下的原始帧号序列。"""
    p = perm_for(ep)
    order: list[int] = []
    order += list(range(BOUND[0], BOUND[1]))       # seg0 恒第一
    for k in p:
        order += list(range(BOUND[k], BOUND[k + 1]))
    order += list(range(BOUND[5], BOUND[6]))       # seg5 收尾
    return order


def cycle_positions(ep: int) -> dict[int, int]:
    """原始帧号 -> 新的 cycle_idx 值（1..6）。"""
    p = perm_for(ep)
    pos: dict[int, int] = {}
    for f in range(BOUND[0], BOUND[1]):
        pos[f] = 1
    for slot, k in enumerate(p, start=2):          # 第 2..5 位
        for f in range(BOUND[k], BOUND[k + 1]):
            pos[f] = slot
    for f in range(BOUND[5], BOUND[6]):
        pos[f] = 6
    return pos


def build_parquet(ep: int, out: Path) -> tuple[int, dict]:
    src = SRC / f"data/chunk-000/episode_{ep:06d}.parquet"
    df = pd.read_parquet(src)

    def col(c):
        x = df[c].to_numpy()
        if x.dtype == object:
            x = np.stack([np.asarray(v, dtype=np.float64) for v in x])
        return np.asarray(x, dtype=np.float64)

    st, ac = col("observation.state"), col("action")
    n = len(st)
    assert n == 2101, f"ep{ep} 帧数不是 2101 而是 {n}"

    order = np.asarray(frame_order(ep), dtype=np.int64)
    assert len(order) == n and sorted(order.tolist()) == list(range(n)), "重排不是双射"

    pos = cycle_positions(ep)
    new_idx = np.asarray([pos[int(f)] for f in order], dtype=np.float64)

    new_st = st[order].copy()
    new_st[:, 21] = new_idx                        # 只改 dim21

    # 保持 action[t] == state[t+1] 逐字节成立
    new_ac = np.empty_like(new_st)
    new_ac[: n - 1] = new_st[1:]
    new_ac[n - 1] = ac[-1].copy()

    fi = np.arange(n, dtype=np.int64)
    out_df = pd.DataFrame({
        "observation.state": list(new_st.astype(np.float32)),
        "action": list(new_ac.astype(np.float32)),
        "timestamp": (fi / FPS).astype(np.float32),
        "frame_index": fi,
        "episode_index": np.full(n, ep, dtype=np.int64),
        "index": (fi + ep * n).astype(np.int64),
        "task_index": df["task_index"].to_numpy().astype(np.int64)[order],
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    # 源 parquet 的 schema metadata 是 None，而 pandas 会塞一个 `pandas` 块进去。
    # 列型/行数完全一致，多余 metadata 一般无害 —— 但源是干净的，就保持一致，
    # 免得起训后出问题还要回头怀疑是不是这里。见 _verify_schema 的对照。
    _strip_meta(out)
    return n, {"perm": list(perm_for(ep)), "order_head": order[:5].tolist(),
               "new_idx_unique": sorted(set(new_idx.tolist()))}


def _strip_meta(path: Path) -> None:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    pq.write_table(t.replace_schema_metadata(None), path)


def build_video(ep: int, cam: str, out: Path) -> int:
    """用 select 滤镜按新帧序输出，帧级精确。"""
    src = SRC / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
    order = frame_order(ep)
    # 把连续的帧号压成 [a,b] 区间，减少表达式项数
    runs: list[tuple[int, int]] = []
    for f in order:
        if runs and f == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], f)
        else:
            runs.append((f, f))
    expr = "+".join(f"between(n\\,{a}\\,{b})" for a, b in runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"select='{expr}'", "-vsync", "0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(cmd, check=True)
    return len(runs)


def probe_frames(p: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "default=nw=1:nk=1", str(p)],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只造前 N 集（验证用）")
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()

    info = json.loads((SRC / "meta/info.json").read_text())
    total = int(info["total_episodes"])
    eps = range(total if a.limit <= 0 else min(a.limit, total))

    # 先清空目标（避免上一次的残渣混进来）
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "meta").mkdir(parents=True)
    (DST / "data/chunk-000").mkdir(parents=True)

    lens, perms_used = [], []
    for ep in eps:
        n, meta = build_parquet(ep, DST / f"data/chunk-000/episode_{ep:06d}.parquet")
        lens.append(n)
        perms_used.append(meta["perm"])
        if not a.skip_video:
            for cam in CAMS:
                build_video(ep, cam, DST / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4")
        print(f"  ep{ep:03d} 帧{n} perm={meta['perm']} idx集合={meta['new_idx_unique']}",
              flush=True)

    # ---- meta ----
    info["total_episodes"] = len(lens)
    info["total_frames"] = int(sum(lens))
    info["total_videos"] = len(lens) * len(CAMS) if not a.skip_video else 0
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False))
    shutil.copy(SRC / "meta/tasks.jsonl", DST / "meta/tasks.jsonl")

    src_eps = [json.loads(x) for x in (SRC / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    with (DST / "meta/episodes.jsonl").open("w") as f:
        for i, ep in enumerate(eps):
            d = dict(src_eps[ep]); d["episode_index"] = ep; d["length"] = lens[i]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # episodes_stats：逐集重算（重排会改 dim21 的统计）
    src_stats = [json.loads(x) for x in (SRC / "meta/episodes_stats.jsonl").read_text().splitlines() if x.strip()]
    with (DST / "meta/episodes_stats.jsonl").open("w") as f:
        for i, ep in enumerate(eps):
            d = src_stats[ep]
            df = pd.read_parquet(DST / f"data/chunk-000/episode_{ep:06d}.parquet")
            for key in ("observation.state", "action"):
                x = np.stack([np.asarray(v, float) for v in df[key].to_numpy()])
                d["stats"][key] = {
                    "min": x.min(0).tolist(), "max": x.max(0).tolist(),
                    "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                    "count": [len(x)],
                }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n✓ 写出 {DST}")
    print(f"  episodes={len(lens)}  frames={sum(lens)}  用到的排列 {len(set(map(tuple,perms_used)))} 种")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
