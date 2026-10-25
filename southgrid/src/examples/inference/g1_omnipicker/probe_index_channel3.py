#!/usr/bin/env python
"""探针 v3：把量尺自身的噪声压下去。

v2 跑出来的「斜率中位 0.293（图+state）/ 0.372（图置零）」**不能用** —— 同一份输出里
噪声底中位是 103.7mm / 117.9mm，和被测量（Δk=19 中位几十 mm）同量级。两个原因：

  毛病 1：参照路和换值路**各只采 1 次**。pi0.5 是随机流匹配采样，单次采样的抖动
    就是上面那个 100mm 量级 ⇒ 量出来的"斜率"主要在量噪声。这是 v1/v2 共有的
    真正致命处（v1 那个 0.09 因此也不可信，只是它还有个分布外输入的毛病挡在前面）。
  毛病 2："噪声底"取的是 max-over-重复 且 max-over-20步，是极值统计量，不能和
     单步的 Δ 直接比 —— 它把噪声报大了。

v3 的做法：
  · 每路采 R 次（默认 10）取**均值轨迹**，噪声按 1/sqrt(R) 掉；
  · 噪声报**每步 sd 的中位**（真噪声尺度，不是极值）；
  · 按「两个独立均值之差」算每步的标准误 SE，给出 t = Δ/SE；
  · 只有 t>2 的对才算量到了信号，主指标是**这些对的斜率中位**；
  · 取帧缓存（同一个 (集,帧号) 在条件 A/B 下是同一帧，v2 解了两遍）。

计划（换成哪一帧、换哪几个维度）完全沿用 v2 的 build_plan，保证 v2/v3 与
训练前后量到的是同一个东西。

用法：
  /home/dgl/lxc/openpi/.venv/bin/python probe_index_channel3.py --run --episodes 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import probe_index_channel2 as P2

_FC = {}

# 换哪几个维度 ⇒ 这个量尺问的是**哪个通道**在起作用。
# 训练时的干预只弄脏 19/20，dim21 始终保持真值；所以：
#   all   = 18:22（整块索引，和训练干预同构）—— 模型整体有多听索引块
#   idx   = 只换 21（cycle_idx 单独被弄脏，19/20 真值）—— 单看"听不听 cycle_idx"
#   label = 只换 19,20（goal_y/cycle_dist 弄脏，cycle_idx 真值）—— 单看"听不听 goal_y"
# 判读（训练后）：
#   idx 高 且 label 低 ⇒ 学到的是 cycle_idx ← 这是目标
#   label 高 且 idx 低 ⇒ 学到的还是 goal_y，干预失败
SWAP = {"all": slice(18, 22), "idx": slice(21, 22), "label": slice(19, 21),
        "none": slice(0, 0)}
DIMS_DOC = {"all": "整块索引（与训练干预同构）",
            "idx": "只脏 cycle_idx，19/20 保持真值",
            "label": "只脏 goal_y/cycle_dist，cycle_idx 保持真值",
            "none": "索引一个都不换（配合 --image don 单独量视觉的拉力）"}

# 图像用谁的 ⇒ 和 --dims 组合成一个 2 维设计，用来把"听索引"和"听视觉"分开：
#   (dims=all,  image=ref) = 只动索引块，图像真值 ⇒ 索引块单独的拉力
#   (dims=none, image=don) = 只动图像，索引真值   ⇒ 视觉单独的拉力
# 两者都跟同一个参照（真值 state + 真值图像）比。训练时图像里工具一直在，
# 所以模型完全可能改用视觉把 dim19 的矛盾解开 —— 光看 loss 分不出这两条路。
IMAGE_DOC = {"ref": "用参照集的图像（真值）", "don": "用供体集的图像"}


def frames(e: int, f: int):
    """(集, 帧号) → (头相机, 腕相机) 的 RGB 帧，缓存。

    条件 A/B 用的是同一帧，v2 把两路视频各解了两遍（每对 1.3s）。
    """
    if (e, f) not in _FC:
        _FC[(e, f)] = (P2.video_frame(e, "cam_head", f),
                       P2.video_frame(e, "cam_wrist_r", f))
    return _FC[(e, f)]


def sample(pol, st, h, w, prompt, r, blank):
    """对同一个输入采 r 次，返回 (r, 20, 3) 的末端 xyz 轨迹。"""
    if blank:
        h = np.zeros_like(h)
        w = np.zeros_like(w)
    chw_h = np.transpose(np.ascontiguousarray(h), (2, 0, 1))
    chw_w = np.transpose(np.ascontiguousarray(w), (2, 0, 1))
    out = []
    for _ in range(r):
        a = np.asarray(pol.infer({"state": st,
                                  "images": {"cam_head": chw_h, "cam_wrist_r": chw_w},
                                  "prompt": prompt})["actions"], dtype=np.float32)
        out.append((a if a.ndim > 1 else a[None])[:20, P2.POS])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10, help="每路采样次数 R")
    ap.add_argument("--tag", default="base", help="结果文件名后缀，如 base / d22idx3k")
    ap.add_argument("--dims", choices=tuple(SWAP), default="all",
                    help="换哪几个索引维（见文件头 SWAP 的判读说明）")
    ap.add_argument("--image", choices=("ref", "don"), default="ref",
                    help="图像用参照集的还是供体集的（配合 --dims none 量视觉的拉力）")
    args = ap.parse_args()
    if not args.run:
        args.plan = True

    plan = P2.build_plan(list(range(args.episodes)), P2.K_LIST, "cross-ep")
    n_all = len(plan)
    plan = [it for it in plan if it["dgoal_mm"] >= 5.0]
    print(f"cross-ep 方案 {n_all} 对，剔除 Δgoal_y<5mm 的 {n_all-len(plan)} 对，"
          f"量斜率用 {len(plan)} 对；每路采 R={args.reps} 次")
    print(f"换的维度 = {args.dims} → {SWAP[args.dims]}；{DIMS_DOC[args.dims]}")

    if not args.run:
        print("（只跑了计划模式）")
        return

    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8010)
    print(f"策略服务器 {pol.get_server_metadata()}\n")

    results = {}
    for cond, blank in (("A 图+state（分布内）", False), ("B 图置零（近似评测）", True)):
        print(f"########## 条件 {cond} ##########")
        print(f"{'集':>3} {'相':>3} {'k':>3} {'位姿距':>7} {'Δ目标':>7} {'噪声sd':>7} "
              f"{'Δk=0':>7} {'Δk=9':>7} {'Δk=19':>7} {'SE19':>6} {'斜率':>6} {'t':>6} {'朝供体':>6}")
        rows = []
        for it in plan:
            S, A, ti = P2.ep(it["ep"])
            S_don, _, _ = P2.ep(it["ep_don"])
            don_l = it["don_glob"] % 2101
            prompt = P2.TASKS[ti]
            # 图像取谁的：参照帧（真值）还是供体帧。供体帧的位姿和参照帧只差 ~7mm，
            # 所以「参照集的 state + 供体集的图像」仍然是个大致自洽的输入。
            if args.image == "don":
                h, w = frames(it["ep_don"], don_l)
            else:
                h, w = frames(it["ep"], it["ref_f"])

            st_ref = S[it["ref_f"]].copy()
            st_sw = S[it["ref_f"]].copy()
            st_sw[SWAP[args.dims]] = S_don[don_l, SWAP[args.dims]]

            R_ref = sample(pol, st_ref, h, w, prompt, args.reps, blank)
            R_sw = sample(pol, st_sw, h, w, prompt, args.reps, blank)
            m_ref, m_sw = R_ref.mean(0), R_sw.mean(0)

            # 每步 sd（两路合并），按轴算再取模长 → 该步的噪声尺度(mm)
            var = 0.5 * (R_ref.var(0, ddof=1) + R_sw.var(0, ddof=1))
            sd_step = np.linalg.norm(np.sqrt(var), axis=-1) * 1000
            # 两个独立均值之差的标准误（每轴先算再取模长）
            se_vec = np.sqrt(R_ref.var(0, ddof=1) / args.reps + R_sw.var(0, ddof=1) / args.reps)
            se_step = np.linalg.norm(se_vec, axis=-1) * 1000

            d = [float(np.linalg.norm(m_sw[kk] - m_ref[kk])) * 1000 for kk in (0, 9, 19)]
            t = d[2] / max(se_step[19], 1e-9)
            lo = lambda f: min(f, 2101 - 20)
            t_ref = A[lo(it["ref_f"]):lo(it["ref_f"]) + 20, P2.POS]
            t_don = S_don[lo(don_l):lo(don_l) + 20, P2.POS]
            toward = (np.linalg.norm(m_sw[19] - t_don[19]) <
                      np.linalg.norm(m_sw[19] - t_ref[19]))
            slope = d[2] / max(it["dgoal_mm"], 1e-9)
            dg = max(it["dgoal_mm"], 1e-9)
            rows.append(dict(ep=it["ep"], idx_p=it["idx_p"], k=it["k"],
                             pose_mm=it["pose_mm"], dgoal_mm=it["dgoal_mm"],
                             noise=float(np.median(sd_step)), sd19=float(sd_step[19]),
                             d=d, se=float(se_step[19]), slope=slope, t=t,
                             w=float(se_step[19]) / dg,      # 斜率的标准误
                             toward=bool(toward)))
            print(f"{it['ep']:>3} {it['idx_p']:>3} {it['k']:>3} {it['pose_mm']:>8.1f} "
                  f"{it['dgoal_mm']:>7.1f} {np.median(sd_step):>7.1f} {d[0]:>7.1f} "
                  f"{d[1]:>7.1f} {d[2]:>7.1f} {se_step[19]:>6.1f} {slope:>6.3f} "
                  f"{t:>6.2f} {str(toward):>6}")
        results[cond] = rows
        sl = np.array([r["slope"] for r in rows])
        tt = np.array([r["t"] for r in rows])
        nz = np.array([r["noise"] for r in rows])
        tg = np.array([r["toward"] for r in rows])
        sig = tt > 2.0
        # 主指标：逆方差加权平均斜率。每对的斜率 = Δ/Δgoal，权重 = 1/(SE/Δgoal)²。
        # 只用 t>2 的对等于把 2/3 的数据扔掉；加权平均把 50 对全用上，
        # 单对噪声按 1/sqrt(50) 继续掉。
        # 主指标：**不加权**均值 ± 对间 SE。
        # 逆方差加权看着更"正确"，但权重 1/(SE/Δgoal)² 里的 SE 是从 R 次采样估出来的，
        # 本身噪声很大（本档实测每对 SE 在 2.3~35.6mm 之间差 15 倍）⇒ 权重被系统性
        # 压给"恰好把自身噪声估小了"的对，SE 被严重低估（t=40 那种数就是这个来的）。
        # 点估计两种算法一致（0.212 / 0.211），所以只用不加权的 SE 做推断。
        w = 1.0 / np.array([r["w"] for r in rows]) ** 2
        mu_w = float((sl * w).sum() / w.sum())
        mu = float(sl.mean())
        mu_se = float(sl.std(ddof=1) / np.sqrt(len(sl)))
        print(f"\n  每步噪声 sd 中位 {np.median(nz):.1f}mm   "
              f"（R={args.reps} 下**第19步**均值差的 SE 中位 {np.median([r['se'] for r in rows]):.1f}mm）")
        print(f"  平均斜率 = {mu:.3f} ± {mu_se:.3f}  (t={mu/max(mu_se,1e-9):.1f}, "
              f"{len(rows)} 对对间 SE)  ← 主指标")
        print(f"  （逆方差加权点估计 {mu_w:.3f}，仅作交叉核对；其 SE 不可信，理由见代码注释）")
        print(f"  斜率中位（全部 {len(rows)} 对）      {np.median(sl):.3f}")
        print(f"  斜率中位（仅 t>2 的 {int(sig.sum())} 对）  "
              f"{np.median(sl[sig]):.3f}" if sig.any() else "  没有一对 t>2 ⇒ 没量到信号")
        print(f"  t>2 的比例 {100.0*sig.mean():.1f}%   输出朝供体移动的比例 {100.0*tg.mean():.1f}%\n")

    a = results["A 图+state（分布内）"]
    b = results["B 图置零（近似评测）"]
    print(f"===== 小结 [{args.tag}] =====")
    summary = {}
    for nm, rows in (("A 图+state", a), ("B 图置零", b)):
        tt = np.array([r["t"] for r in rows])
        sl = np.array([r["slope"] for r in rows])
        ww = 1.0 / np.array([r["w"] for r in rows]) ** 2
        mu_w = float((sl * ww).sum() / ww.sum())
        mu = float(sl.mean())
        mu_se = float(sl.std(ddof=1) / np.sqrt(len(sl)))
        sig = tt > 2.0
        summary[nm] = dict(
            mean_slope=mu, mean_slope_se=mu_se, mean_slope_t=mu / max(mu_se, 1e-9),
            weighted_slope_point=mu_w,   # 只作交叉核对，SE 不可信
            n_pairs=len(rows), n_t_gt2=int(sig.sum()),
            median_slope_all=float(np.median(sl)),
            median_slope_t2=float(np.median(sl[sig])) if sig.any() else None,
            median_noise_sd=float(np.median([r["noise"] for r in rows])),
            toward_rate=float(np.mean([r["toward"] for r in rows])),
            n_negative=int((sl < 0).sum()))
        s = summary[nm]
        print(f"  {nm:<12} 噪声sd {s['median_noise_sd']:>6.1f}mm  "
              f"平均斜率 {mu:.3f}±{mu_se:.3f} (t={s['mean_slope_t']:.1f})  "
              f"t>2 {s['n_t_gt2']:>2}/{len(rows)}  朝供体 {100*s['toward_rate']:.0f}%  "
              f"斜率为负 {s['n_negative']} 对")

    # 文件名保持向后兼容：image=ref 时还是 probe_v3_<tag>_<dims>.json
    suffix = args.dims if args.image == "ref" else f"{args.dims}_imgdon"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"probe_v3_{args.tag}_{suffix}.json")
    with open(out, "w") as f:
        import json
        json.dump(dict(tag=args.tag, dims=args.dims, image=args.image,
                       episodes=args.episodes, reps=args.reps, n_pairs=len(plan),
                       summary=summary, rows=results), f,
                  ensure_ascii=False, indent=1)
    print(f"\n逐对明细已落盘：{out}")


if __name__ == "__main__":
    main()
