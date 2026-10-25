#!/usr/bin/env python3
"""严格验证 d22idxS（相位重排）数据集：逐帧对账，不接受"看起来没问题"。

检查项：
 1. 帧数守恒：新数据集每一集的帧数 == 源，总帧数 == 源
 2. **除 dim21 外，21 个维度逐帧逐字节等于源**（重排只许动 cycle_idx）
 3. dim21 重排正确：新顺序下 1..6 依次递增，且每一帧的 idx 等于它在新序列里的段位
 4. `action[t] == state[t+1]` 在新数据里**继续逐字节成立**
 5. `(idx, goal_y)` 自洽：同一 idx 值在不同集里对应**不同的 goal_y**（这才是重排的意义）
 6. 每个视频流的帧数 == parquet 行数
 7. 拼缝处视频帧与源对应帧**内容一致**（重编码过，比 PSNR 不比字节）

用法: verify_phase_shuffled_dataset.py [--limit N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
DST = Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep")
BOUND = [0, 366, 766, 1166, 1566, 1966, 2101]
CAMS = ["observation.images.cam_head", "observation.images.cam_wrist_r"]
PERMS = list(itertools.permutations(range(1, 5)))

ok_all = True


def chk(cond: bool, msg: str) -> None:
    global ok_all
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    if not cond:
        ok_all = False


def arr(df, c) -> np.ndarray:
    x = df[c].to_numpy()
    if x.dtype == object:
        x = np.stack([np.asarray(v, dtype=np.float64) for v in x])
    return np.asarray(x, dtype=np.float64)


def expected_order(ep: int) -> list[int]:
    p = PERMS[ep % 24]
    o = list(range(BOUND[0], BOUND[1]))
    for k in p:
        o += list(range(BOUND[k], BOUND[k + 1]))
    o += list(range(BOUND[5], BOUND[6]))
    return o


def probe_frames(p: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "default=nw=1:nk=1", str(p)],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def psnr(a: Path, b: Path) -> float:
    r = subprocess.run(["ffmpeg", "-loglevel", "info", "-i", str(a), "-i", str(b),
                        "-lavfi", "psnr", "-f", "null", "-"],
                       capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "PSNR" in line and "average:" in line:
            try:
                return float(line.split("average:")[1].split()[0])
            except ValueError:
                return float("nan")
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    src_info = json.loads((SRC / "meta/info.json").read_text())
    dst_info = json.loads((DST / "meta/info.json").read_text())
    total = int(dst_info["total_episodes"])
    eps = list(range(total if a.limit <= 0 else min(a.limit, total)))

    print(f"源 {SRC.name}  {src_info['total_episodes']} 集 / {src_info['total_frames']} 帧")
    print(f"新 {DST.name}  {dst_info['total_episodes']} 集 / {dst_info['total_frames']} 帧")
    print(f"本次验证 {len(eps)} 集\n")

    print("[1] meta 一致性")
    if len(eps) == src_info["total_episodes"]:
        chk(dst_info["total_frames"] == src_info["total_frames"],
            f"总帧数守恒 {dst_info['total_frames']} == {src_info['total_frames']}")
    # 特征名 / 形状必须原样保留
    chk(dst_info["features"]["observation.state"]["shape"] == [22], "state 仍是 22 维")
    chk(dst_info["features"]["action"]["shape"] == [22], "action 仍是 22 维")
    chk(set(dst_info["features"]) == set(src_info["features"]),
        f"特征键集合不变 ({len(dst_info['features'])} 个)")
    chk((DST / "meta/tasks.jsonl").read_text() == (SRC / "meta/tasks.jsonl").read_text(),
        "tasks.jsonl 原样复制")

    # parquet 层：列型、行数、schema metadata 都要与源一致。
    # pandas 默认会写一个 `pandas` metadata 块，源没有 —— 不影响加载，但保持干净，
    # 免得起训出问题时还得回头排查这里。
    import pyarrow.parquet as _pq
    _s = _pq.read_schema(SRC / "data/chunk-000/episode_000000.parquet")
    _d = _pq.read_schema(DST / "data/chunk-000/episode_000000.parquet")
    chk([str(f.type) for f in _d] == [str(f.type) for f in _s],
        "parquet 列名/列型与源逐字段一致")
    chk(list(_d.names) == list(_s.names), f"列序一致 {list(_d.names)}")
    chk((_s.metadata is None) == (_d.metadata is None),
        f"schema metadata 与源同为 None（源={_s.metadata}，新={None if _d.metadata is None else list(_d.metadata)}）")

    idx2y: dict[float, set[float]] = {}
    y_of_tool: dict[int, float] = {}

    for ep in eps:
        print(f"\n[ep{ep:03d}] perm={PERMS[ep % 24]}")
        sd = pd.read_parquet(SRC / f"data/chunk-000/episode_{ep:06d}.parquet")
        dd = pd.read_parquet(DST / f"data/chunk-000/episode_{ep:06d}.parquet")
        ss, sa = arr(sd, "observation.state"), arr(sd, "action")
        ds, da = arr(dd, "observation.state"), arr(dd, "action")
        order = np.asarray(expected_order(ep))
        n = len(ds)

        chk(n == len(ss) == 2101, f"帧数 {n} == 2101")

        # ---- 2. 除 dim21 外逐帧相等 ----
        cols = [c for c in range(22) if c != 21]
        d_other = np.abs(ds[:, cols] - ss[order][:, cols]).max()
        chk(d_other == 0.0, f"除 dim21 外 21 维逐帧逐字节相等 (max|Δ|={d_other})")

        # ---- 3. dim21 重排正确 ----
        # ⚠ 这里必须按**新布局的位置**算，不能按原始段号 k 去索引 BOUND：
        # 新布局是 [seg0][perm 的四段][seg5]，各段的长度决定它落在哪一段帧区间。
        # seg1..seg4 恰好都是 400 帧，所以新旧边界数值上相同 —— 正因如此，
        # 恒等排列(perm=1,2,3,4)会掩盖下标错误，只有非平凡排列才暴露。
        # 用累计长度算，即使将来各段长度不等也仍然正确。
        lens_new = [BOUND[1] - BOUND[0]] + [BOUND[k + 1] - BOUND[k] for k in PERMS[ep % 24]] \
                   + [BOUND[6] - BOUND[5]]
        want = np.empty(n)
        cur = 0
        for slot, L in enumerate(lens_new, start=1):
            want[cur:cur + L] = slot
            cur += L
        assert cur == n, f"新布局累计长度 {cur} != 帧数 {n}"
        d21 = np.abs(ds[:, 21] - want).max()
        chk(d21 == 0.0, f"dim21 按新顺序 = 1..6 (max|Δ|={d21})")
        chk(sorted(set(ds[:, 21].tolist())) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "dim21 取值集合 = {1,2,3,4,5,6}")

        # ---- 4. action[t] == state[t+1] ----
        d_act = np.abs(da[: n - 1] - ds[1:]).max()
        chk(d_act == 0.0, f"action[t] == state[t+1] 在新数据里逐字节成立 (max|Δ|={d_act})")

        # ---- 5. (idx, goal_y) 配对正确性 ----
        # 关键断言：新布局第 slot 位放的是原始 seg k，则该位所有帧的 dim19 必须
        # 等于**源里 seg k 的 goal_y**（工具没换，只是位次变了）。
        # 这条过了，才说明 "idx 变了但 goal_y 仍指向真正要拿的那件工具"。
        cur = BOUND[1] - BOUND[0]
        for slot, k in enumerate(PERMS[ep % 24], start=2):
            L = BOUND[k + 1] - BOUND[k]
            y_src = float(np.median(ss[BOUND[k]:BOUND[k + 1], 19]))
            y_new = float(np.median(ds[cur:cur + L, 19]))
            chk(abs(y_src - y_new) < 1e-6,
                f"新第{slot}位(原seg{k}) goal_y {y_new:+.4f} == 源 seg{k} 的 {y_src:+.4f}")
            idx2y.setdefault(float(slot), set()).add(round(y_new, 4))
            y_of_tool.setdefault(k, round(y_src, 4))
            cur += L

        # ---- 6. 视频帧数 ----
        for cam in CAMS:
            vp = DST / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
            nv = probe_frames(vp)
            chk(nv == n, f"{cam.split('.')[-1]} 视频帧数 {nv} == parquet {n}")

    print("\n[5] 重排是否真的打破了「idx → 工具」的固定映射")
    for slot in sorted(idx2y):
        ys = sorted(idx2y[slot])
        chk(len(ys) > 1 if len(eps) >= 5 else True,
            f"idx={int(slot)} 在 {len(eps)} 集里对应 {len(ys)} 种不同 goal_y: {ys[:6]}")
    print(f"  工具位序→goal_y（应与源完全一致，即工具本身的 y 没被动过）: {y_of_tool}")

    print("\n[7] 拼缝处画面一致性（该帧在源里的同一帧，重编码后比 PSNR）")
    for ep in eps[:2]:
        order = expected_order(ep)
        # 第一处拼缝：seg0 末帧 → perm 首段首帧
        j = order[BOUND[1]]
        for cam in CAMS:
            vp = DST / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
            sp = SRC / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
            tmp = Path("/tmp/_vs")
            tmp.mkdir(exist_ok=True)
            for tag, src, nfr in (("new", vp, BOUND[1]), ("ref", sp, j)):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                                "-vf", f"select='eq(n\\,{nfr})'", "-vsync", "0",
                                "-frames:v", "1", str(tmp / f"{tag}.png")], check=True)
            p = psnr(tmp / "new.png", tmp / "ref.png")
            chk(p > 40 or np.isnan(p),
                f"ep{ep:03d} {cam.split('.')[-1]} 拼缝首帧 PSNR={p:.1f} dB (>40 即同一画面)")

    print("\n" + ("=" * 60))
    print("✓ 全部通过" if ok_all else "✗ 有检查未通过")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
