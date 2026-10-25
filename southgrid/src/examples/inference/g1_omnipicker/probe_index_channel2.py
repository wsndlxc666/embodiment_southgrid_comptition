#!/usr/bin/env python
"""探针 v2：模型到底有多依赖 state 里的周期索引（dim18~21）？

`probe_index_channel.py`（v1）的两个毛病，v2 都改掉：

  毛病 1（致命）：v1 把图像和臂位姿**全冻结在帧 0**（home 位姿），只换 dim18:22。
    但帧 0 的 home 位姿在训练里**只和 goal_y=工具1 一起出现过**（phase_pose_diag.py：
    ph0 云心离 ph1~ph4 有 660mm），"home 位姿 + 工具5 的目标"这个组合训练集里不存在。
    模型看到的是分布外输入 ⇒ 量出来的 0.09 是**输入的锅**，不是"模型不用索引"。
  毛病 2：只量 `chunk[0]`（chunk 的第一步）。该去第几件工具在"下一步"上几乎看不出来 ——
    手臂在 home 起步，朝工具1 和朝工具5 的第一小步几乎一样。

v2 的做法：
  · 用**真实边界帧**（变化点 366/766/1166/1566 之后 k 帧）自己的图像和位姿；
  · 换进去的索引值来自**同一个集、同一个相位内偏移**的另一个相位 —— 值域完全真实；
  · 而且挑**位姿最接近**的那个相位来换（cross-phase 云重叠最重的那一对），
    这是对模型最有利、也最考验索引通道的一种换法；
  · 量**整个 chunk**（k=0/9/19）以及"是否朝换过去的那个相位移动"。

两种条件：
  A. 图 + state 都给（分布内）—— 这时视觉也能定相位，测的是"索引的相对权柄"
  B. 图置零（近似评测：实测视觉在评测里是死的）—— 测的是"只剩 state 文字时靠不靠索引"

先用 `--plan` 跑（**不需要策略服务器**）：打印每一对换法的位姿距离、换后状态到最近真实
训练帧的距离，确认输入确实在分布内。确认后再 `--run` 起服务器量。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_index_channel2.py --plan
  （策略服务器在 8010 之后）/home/dgl/lxc/openpi/.venv/bin/python probe_index_channel2.py --run
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# 参照集。**默认是原始 d22**（idx 是帧号的纯函数那张表）。
# ⚠ 探针 idxS 档时要注意：idxS 模型**恰恰是被训练成不信这张表的**
#   （同一个 idx 值在别集里对应别的工具）。用原始布局的帧去探它，
#   测的是"它在原布局下会不会照表走"——那正是它被训练掉的行为，
#   所以读数会**偏向"没变"**，对照臂 idxR/11999 反而不吃亏。
#   想正面问"图像与 idx 打架时它听谁"，要把参照集也换成 idxS：
#       PROBE_DS=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep ...
#   但那时 CHANGE 那张表也不再成立（每集排列不同），要另行设计 —— 别直接套用。
DS = os.environ.get("PROBE_DS", "/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
CAM = {"cam_head": "observation.images.cam_head",
       "cam_wrist_r": "observation.images.cam_wrist_r"}
TASKS = {0: "把工具全部收进箱子里",
         1: "把桌面上的工具都收进工具箱",
         2: "把所有工具从左到右放回工具箱"}
# 变化点（新相位首帧）与其 cycle_idx —— **只在 DS 是原始 d22 时成立**
CHANGE = [(366, 2), (766, 3), (1166, 4), (1566, 5)]
K_LIST = [0, 5, 10, 20, 40]
POS = slice(7, 10)          # 右末端 xyz（base 系，米）
IDX_DIMS = slice(18, 22)    # goal_dist / goal_y / cycle_dist / cycle_idx

_S = {}


def ep(e):
    if e not in _S:
        import pandas as pd
        df = pd.read_parquet(f"{DS}/data/chunk-000/episode_{e:06d}.parquet",
                             columns=["observation.state", "action", "task_index"])
        _S[e] = (np.array([list(s) for s in df["observation.state"]], dtype=np.float64),
                 np.array([list(a) for a in df["action"]], dtype=np.float64),
                 int(df["task_index"].iloc[0]))
    return _S[e]


def all_states():
    """全数据集的 state（用来判"换完之后还在不在分布内"）。"""
    if "all" not in _S:
        import pyarrow.parquet as pq
        import glob
        fs = sorted(glob.glob(os.path.join(DS, "data", "chunk-*", "episode_*.parquet")))
        _S["all"] = np.concatenate(
            [np.stack([np.asarray(v, dtype=np.float64)
                       for v in pq.read_table(f, columns=["observation.state"])["observation.state"].to_pylist()])
             for f in fs])
    return _S["all"]


def video_frame(e, cam, idx):
    import av
    p = f"{DS}/videos/chunk-000/{CAM[cam]}/episode_{e:06d}.mp4"
    with av.open(p) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i == idx:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(idx)


def phase_table():
    """全数据集的 (相位号 3..6, 相位内偏移)。

    所有集的变化点逐字节一致（verify_cycle_index 已校验），所以直接按统一帧号切。
    """
    key = "phase"
    if key not in _S:
        starts = [0, 366, 766, 1166, 1566, 1966]
        n = all_states().shape[0]
        per = n // 100
        ph = np.zeros(n, dtype=np.int32)
        off = np.zeros(n, dtype=np.int32)
        for t in range(per):
            p = max(i for i, c in enumerate(starts) if c <= t)
            ph[t::per] = p
            off[t::per] = t - starts[p]
        _S[key] = (ph, off)
    return _S[key]


def build_plan(episodes, k_list, mode):
    """每个 (集, 相位, k) 配一个"供体"帧，只取它的 dim18~21。

    mode="cross-ep"：在**全部 100 集**里找位姿最接近的异相位帧，且要求相位内偏移 k 相同。
      同集内异相位位姿至少差 26mm（实测），跨集抖动才会把云凑到一起 ——
      所以"位姿几乎一样、目标差一整个工具"的输入只能是跨集构造的。这才是真歧义输入。
    mode="same-ep" ：只在同一集里找 —— 对照组。位姿至少差 26mm，模型看位姿就能定相位，
      完全可以不理索引；两组的斜率差就是要量的东西。
    """
    S_all = all_states()
    ph, off = phase_table()
    plan = []
    for e in episodes:
        S, A, _ = ep(e)
        for p, (c_p, idx_p) in enumerate(CHANGE):
            phase_p = p + 1                      # 相位号：变化点 366 → 相位 1
            for k in k_list:
                ref_f = c_p + k
                pose = S[ref_f, POS]
                # 候选：异相位 且 相位内偏移相同
                cand = (ph != phase_p) & (off == k)
                if mode == "same-ep":
                    ep_idx = np.zeros(S_all.shape[0], dtype=bool)
                    ep_idx[e * 2101 : (e + 1) * 2101] = True
                    cand &= ep_idx
                cs = np.flatnonzero(cand)
                d = np.linalg.norm(S_all[cs, POS] - pose, axis=-1) * 1000
                j = int(cs[int(d.argmin())])
                e_don = j // 2101
                plan.append(dict(ep=e, p=p, k=k, ref_f=ref_f, don_glob=j, ep_don=e_don,
                                 idx_p=idx_p, idx_q=int(round(S_all[j, 21])),
                                 pose_mm=float(d.min()),
                                 dgoal_mm=abs(S[ref_f, 19] - S_all[j, 19]) * 1000))
    return plan


def donors_are_indist(plan):
    """供体帧必须是**真实训练帧**：它的图像/位姿/索引全都存在于数据集里。

    这条是探针有效性的底线 —— v1 那个 0.09 就是因为喂了训练集里不存在的组合。
    这里顺便报：供体帧所在集的**索引维**与参照帧的索引维是否真的不同（不同才有信息）。
    """
    S_all = all_states()
    rows = []
    for it in plan:
        S_ref, _, _ = ep(it["ep"])
        st = S_ref[it["ref_f"]].copy()
        st[IDX_DIMS] = S_all[it["don_glob"], IDX_DIMS]
        # 换成之后的 state 与**供体帧自身**逐维比：位置应该差 pose_mm（这是设计），
        # 索引维必须逐字节相同（否则说明索引被别的东西污染了）
        didx = float(np.abs(st[IDX_DIMS] - S_all[it["don_glob"], IDX_DIMS]).max())
        # 索引维和参照帧差多少 —— 0 就说明这次换法根本没换出信息
        dref = float(np.abs(st[IDX_DIMS] - S_ref[it["ref_f"], IDX_DIMS]).max())
        rows.append((didx, dref))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="真的调策略服务器")
    ap.add_argument("--plan", action="store_true", help="只打印方案与分布内性检查")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--noise", type=int, default=6, help="噪声底重复次数")
    ap.add_argument("--mode", choices=("cross-ep", "same-ep", "both"), default="both")
    args = ap.parse_args()
    if not args.run:
        args.plan = True

    modes = ("cross-ep", "same-ep") if args.mode == "both" else (args.mode,)
    plans = {}
    for mode in modes:
        plan = build_plan(list(range(args.episodes)), K_LIST, mode)
        plans[mode] = plan
        pm = np.array([it["pose_mm"] for it in plan])
        dg = np.array([it["dgoal_mm"] for it in plan])
        print(f"\n===== mode={mode}：{args.episodes} 集 x {len(CHANGE)} 相位 x "
              f"{len(K_LIST)} 个 k = {len(plan)} 对 =====")
        print(f"  参照帧与供体帧的**位置**距离：中位 {np.median(pm):.1f}mm  "
              f"最小 {pm.min():.1f}  最大 {pm.max():.1f}")
        print(f"  两者的目标差 |Δgoal_y|：中位 {np.median(dg):.1f}mm  "
              f"最小 {dg.min():.1f}  最大 {dg.max():.1f}")
        print(f"  位置距离 <10mm 的对数：{int((pm < 10).sum())}/{len(pm)} "
              f"= {100.0*(pm < 10).mean():.1f}%  ← 「位姿几乎一样、目标差一整个工具」")
        di = donors_are_indist(plan)
        print(f"  供体帧逐字节真实：索引维与供体帧自身的最大差 {di[:,0].max():.2e}"
              f"（应恒为 0）；与参照帧的索引差中位 {np.median(di[:,1]):.3f}（>0 才有信息）")
        nodiff = int((dg < 5.0).sum())
        if nodiff:
            print(f"  ⚠ 其中 {nodiff} 对的 Δgoal_y≈0（供体落在 idx=6「全放完」那个相位，"
                  f"goal_y 与上一相位相同）⇒ 没有信息，量斜率时必须剔除")
        if mode == "cross-ep":
            print("  ⇒ 位姿距离那列小，就说明换完之后的 state 在训练集里**真实存在**"
                  "（供体帧自己就是），v1 那种分布外输入的问题没有了。")

    if not args.run:
        print("\n（只跑了计划模式。确认上面的分布内性之后，起策略服务器再加 --run）")
        return
    # 剔掉"换了但目标没变"的对（斜率的分母是 0）
    plan = [it for it in plans["cross-ep"] if it["dgoal_mm"] >= 5.0]
    print(f"\n量斜率用 cross-ep 方案，剔除 Δgoal_y<5mm 的对后剩 {len(plan)} 对")

    # ---------------- 真的调模型 ----------------
    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8010)
    print(f"\n策略服务器 {pol.get_server_metadata()}")

    def infer(st, h, w, prompt):
        chw_h = np.transpose(np.ascontiguousarray(h), (2, 0, 1))
        chw_w = np.transpose(np.ascontiguousarray(w), (2, 0, 1))
        a = np.asarray(pol.infer({"state": st, "images": {"cam_head": chw_h, "cam_wrist_r": chw_w},
                                  "prompt": prompt})["actions"], dtype=np.float32)
        return a if a.ndim > 1 else a[None]

    for cond, blank in (("A 图+state（分布内）", False), ("B 图置零（近似评测）", True)):
        print(f"\n########## 条件 {cond} ##########")
        print(f"{'集':>3} {'相位':>4} {'k':>3} {'位姿距':>7} {'Δ目标':>7} "
              f"{'噪声底':>7} {'Δk=0':>7} {'Δk=9':>7} {'Δk=19':>7} {'斜率':>6} {'朝供体?':>7}")
        slopes, hits, floors = [], [], []
        for it in plan:
            S, A, ti = ep(it["ep"])
            S_don, A_don, _ = ep(it["ep_don"])
            don_l = it["don_glob"] % 2101
            prompt = TASKS[ti]
            h = video_frame(it["ep"], "cam_head", it["ref_f"])
            w = video_frame(it["ep"], "cam_wrist_r", it["ref_f"])
            if blank:
                h = np.zeros_like(h); w = np.zeros_like(w)
            st_ref = S[it["ref_f"]].copy()
            st_sw = S[it["ref_f"]].copy()
            st_sw[IDX_DIMS] = S_don[don_l, IDX_DIMS]

            o_ref = infer(st_ref, h, w, prompt)
            reps = [infer(st_ref, h, w, prompt)[:20, POS] for _ in range(args.noise)]
            floor = float(np.linalg.norm(np.array(reps) - reps[0], axis=-1).max()) * 1000
            o_sw = infer(st_sw, h, w, prompt)

            d = [float(np.linalg.norm(o_sw[kk, POS] - o_ref[kk, POS])) * 1000 for kk in (0, 9, 19)]
            # 换完之后，输出是不是更靠近"供体那个相位"的真实轨迹
            lo = lambda f: min(f, 2101 - 20)
            t_ref = A[lo(it["ref_f"]) : lo(it["ref_f"]) + 20, POS]
            t_don = S_don[lo(don_l) : lo(don_l) + 20, POS]
            toward = (np.linalg.norm(o_sw[19, POS] - t_don[19]) <
                      np.linalg.norm(o_sw[19, POS] - t_ref[19]))
            slope = d[2] / max(it["dgoal_mm"], 1e-9)
            slopes.append(slope); hits.append(toward); floors.append(floor)
            print(f"{it['ep']:>3} {it['idx_p']:>4} {it['k']:>3} {it['pose_mm']:>8.1f} "
                  f"{it['dgoal_mm']:>7.1f} {floor:>7.1f} {d[0]:>7.1f} {d[1]:>7.1f} "
                  f"{d[2]:>7.1f} {slope:>6.3f} {str(toward):>7}")
        print(f"\n  斜率中位 {np.median(slopes):.3f}   噪声底中位 {np.median(floors):.1f}mm   "
              f"输出朝供体移动的比例 {100.0*np.mean(hits):.1f}%")
        print(f"  对照：v1 用帧0/home 位姿量出来的斜率是 0.09~0.12，噪声底见上")


if __name__ == "__main__":
    main()
