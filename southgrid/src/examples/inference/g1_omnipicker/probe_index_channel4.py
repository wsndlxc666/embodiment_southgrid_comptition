#!/usr/bin/env python
"""探针 v4：把 v3 那个**恒正**的主指标换成带符号的投影。

v3 报的 `slope = ‖ΔEE‖/Δgoal` 是个**范数**。v3 自己的对照组戳穿了它：
  (dims=none, image=don) 在条件 B（图置零）下，state 一维没换、两路图像又都被置零
  ⇒ 两次采样是**同分布的独立样本**，Δ 的数学期望就是 0。
  可它量出来 0.227 ± 0.036。
原因：零假设下范数不趋于 0，而趋于「噪声模长 / Δgoal」这个折叠正态的正值地板
（≈ 0.23）。所以 v3 报的基线 all 0.230 / idx 0.219 / label 0.185 全部贴在地板上，
**分不出信号**。v3 里唯一还带符号的是 toward 率（真空 36% / all 66%）。

v4 的主指标：把输出位移**投影到「供体真值轨迹 − 参照真值轨迹」的方向**上。

    u(kk)  = t_don[kk] − t_ref[kk]            # 两个相位的真值 EE 轨迹之差
    proj   = ⟨m_sw[kk] − m_ref[kk], û(kk)⟩     # 有符号，mm
    frac   = proj / ‖u(kk)‖                    # 0 = 完全没听索引；1 = 完全跟到供体轨迹

零假设下 frac 的期望**严格为 0**（不是某个正地板），所以可以直接和 0 比。
内建自检：(dims=none, image=ref) 时两路输入逐字节相同，frac 必须落在 0±噪声内。

保留项：toward 率（带符号、好解读）、‖ΔEE‖（只作历史对照，已知有正地板）、
每步噪声 sd、每条轨迹差 ‖u‖。

用法：
  python probe_index_channel4.py --run --episodes 3 --reps 6 --tag d22idx3k --dims all
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import probe_index_channel2 as P2

_FC = {}

SWAP = {"all": slice(18, 22), "idx": slice(21, 22), "label": slice(19, 21),
        "none": slice(0, 0)}
DIMS_DOC = {"all": "整块索引（与训练干预同构）",
            "idx": "只脏 cycle_idx(21)，19/20 保持真值",
            "label": "只脏 goal_y/cycle_dist(19,20)，cycle_idx 保持真值",
            "none": "索引一个都不换（配 image=ref 就是真空对照）"}
IMAGE_DOC = {"ref": "参照集的图像（真值）", "don": "供体集的图像"}

# 轨迹差太小的对，frac 的分母趋于 0 ⇒ 噪声被放大。
# 但只用来避免除零：**主指标用 proj(mm) 而不是 frac**，proj 不涉及除法，
# 小 ‖u‖ 的对不会放大噪声，所以阈值放得很低（20mm 会砍掉一半样本，别设那么高）。
MIN_U_MM = 5.0


def frames(e: int, f: int):
    if (e, f) not in _FC:
        _FC[(e, f)] = (P2.video_frame(e, "cam_head", f),
                       P2.video_frame(e, "cam_wrist_r", f))
    return _FC[(e, f)]


def sample(pol, st, h, w, prompt, r, blank):
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
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--dims", choices=tuple(SWAP), default="all")
    ap.add_argument("--image", choices=("ref", "don"), default="ref")
    args = ap.parse_args()

    plan = P2.build_plan(list(range(args.episodes)), P2.K_LIST, "cross-ep")
    n0 = len(plan)
    plan = [it for it in plan if it["dgoal_mm"] >= 5.0]
    n1 = len(plan)
    print(f"cross-ep {n0} 对 → 剔除 Δgoal_y<5mm 后 {n1} 对")
    print(f"dims={args.dims}（{DIMS_DOC[args.dims]}）  image={args.image}"
          f"（{IMAGE_DOC[args.image]}）  R={args.reps}")

    if not args.run:
        print("（只跑了计划模式，加 --run）")
        return

    from openpi_client import websocket_client_policy
    pol = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8010)
    print(f"策略服务器 {pol.get_server_metadata()}\n")

    results = {}
    for cond, blank in (("A 图+state（分布内）", False), ("B 图置零（近似评测）", True)):
        print(f"########## 条件 {cond} ##########")
        print(f"{'集':>3} {'相':>3} {'k':>3} {'位姿距':>7} {'‖u19‖':>7} {'噪声sd':>7} "
              f"{'frac19':>7} {'SE':>6} {'z':>6} {'‖Δ19‖':>7} {'朝供体':>6}")
        rows = []
        for it in plan:
            S, A, ti = P2.ep(it["ep"])
            S_don, A_don, _ = P2.ep(it["ep_don"])
            don_l = it["don_glob"] % 2101
            prompt = P2.TASKS[ti]
            h, w = (frames(it["ep_don"], don_l) if args.image == "don"
                    else frames(it["ep"], it["ref_f"]))

            st_ref = S[it["ref_f"]].copy()
            st_sw = S[it["ref_f"]].copy()
            st_sw[SWAP[args.dims]] = S_don[don_l, SWAP[args.dims]]

            R_ref = sample(pol, st_ref, h, w, prompt, args.reps, blank)
            R_sw = sample(pol, st_sw, h, w, prompt, args.reps, blank)
            m_ref, m_sw = R_ref.mean(0), R_sw.mean(0)

            var = 0.5 * (R_ref.var(0, ddof=1) + R_sw.var(0, ddof=1))
            sd_step = np.linalg.norm(np.sqrt(var), axis=-1) * 1000
            se_vec = np.sqrt(R_ref.var(0, ddof=1) / args.reps
                             + R_sw.var(0, ddof=1) / args.reps)     # (20,3) 每轴、mm
            se_vec_mm = se_vec * 1000

            # 真值轨迹两边都取 **action**：模型的输出就是 action，同空间。
            # （A[t] 对 S[t] 中位差 4.0mm、对 S[t+1] 差 7.8mm ⇒ 同帧对齐但有 ~4mm 跟踪残差；
            #   ‖u‖ 才 21mm，一边 action 一边 state 会把这 4mm 混进 u 的方向里。）
            lo = lambda f: min(f, 2101 - 20)
            t_ref = A[lo(it["ref_f"]):lo(it["ref_f"]) + 20, P2.POS]
            t_don = A_don[lo(don_l):lo(don_l) + 20, P2.POS]

            kk = 19
            u = t_don[kk] - t_ref[kk]
            nu = float(np.linalg.norm(u)) * 1000
            if nu < MIN_U_MM:
                continue
            u_hat = u / np.linalg.norm(u)

            delta = (m_sw[kk] - m_ref[kk]) * 1000
            proj = float(np.dot(delta, u_hat))
            se_proj = float(np.linalg.norm(se_vec_mm[kk] * u_hat))
            frac = proj / nu
            se_frac = se_proj / nu
            z = frac / max(se_frac, 1e-9)

            toward = bool(np.linalg.norm(m_sw[kk] - t_don[kk]) <
                          np.linalg.norm(m_sw[kk] - t_ref[kk]))
            rows.append(dict(ep=it["ep"], idx_p=it["idx_p"], k=it["k"],
                             pose_mm=it["pose_mm"], dgoal_mm=it["dgoal_mm"],
                             u_mm=nu, noise=float(np.median(sd_step)),
                             frac=frac, se_frac=se_frac, z=z,
                             proj_mm=proj, norm_mm=float(np.linalg.norm(delta)),
                             slope_norm=float(np.linalg.norm(delta)) / max(it["dgoal_mm"], 1e-9),
                             toward=toward))
            print(f"{it['ep']:>3} {it['idx_p']:>3} {it['k']:>3} {it['pose_mm']:>8.1f} "
                  f"{nu:>7.1f} {np.median(sd_step):>7.1f} {frac:>+7.3f} {se_frac:>6.3f} "
                  f"{z:>6.2f} {np.linalg.norm(delta):>7.1f} {str(toward):>6}")
        results[cond] = rows
        pr = np.array([r["proj_mm"] for r in rows])
        fr = np.array([r["frac"] for r in rows])
        tg = np.array([r["toward"] for r in rows])
        pm, ps = float(pr.mean()), float(pr.std(ddof=1) / np.sqrt(len(pr)))
        mu, mu_se = float(fr.mean()), float(fr.std(ddof=1) / np.sqrt(len(fr)))
        u_med = float(np.median([r["u_mm"] for r in rows]))
        print(f"\n  **平均 proj = {pm:+.2f} ± {ps:.2f} mm**  (t={pm/max(ps,1e-9):+.1f}, "
              f"{len(rows)} 对)   ← 主指标，零假设严格为 0")
        print(f"  参照：日志里 ‖u19‖ 中位 {u_med:.1f}mm —— proj 到这个数的几成就是"
              f"「跟了索引多少」：{pm/max(u_med,1e-9):+.0%}")
        print(f"  （附注 frac = {mu:+.3f}±{mu_se:.3f}，t={mu/max(mu_se,1e-9):+.1f}；"
              f"proj>0 的对数 {int((pr>0).sum())}/{len(pr)} = {100*(pr>0).mean():.0f}%；"
              f"朝供体率 {100*tg.mean():.0f}%）")
        print(f"  噪声sd 中位 {np.median([r['noise'] for r in rows]):.1f}mm   "
              f"（历史口径 ‖Δ‖/Δgoal 均值 {np.mean([r['slope_norm'] for r in rows]):.3f}"
              f"，已知有 ~0.23 正地板，别用）\n")

    summary = {}
    for cond, rows in results.items():
        pr = np.array([r["proj_mm"] for r in rows])
        fr = np.array([r["frac"] for r in rows])
        pm, ps = float(pr.mean()), float(pr.std(ddof=1) / np.sqrt(len(pr)))
        mu, mu_se = float(fr.mean()), float(fr.std(ddof=1) / np.sqrt(len(fr)))
        summary[cond] = dict(
            mean_proj_mm=pm, mean_proj_se=ps, mean_proj_t=pm / max(ps, 1e-9),
            mean_frac=mu, mean_frac_se=mu_se, mean_frac_t=mu / max(mu_se, 1e-9),
            n_pairs=len(rows),
            proj_pos_rate=float((pr > 0).mean()),
            frac_pos_rate=float((fr > 0).mean()),
            toward_rate=float(np.mean([r["toward"] for r in rows])),
            median_noise_sd=float(np.median([r["noise"] for r in rows])),
            median_u_mm=float(np.median([r["u_mm"] for r in rows])),
            mean_slope_norm=float(np.mean([r["slope_norm"] for r in rows])))

    print(f"===== 小结 [{args.tag} / dims={args.dims} / image={args.image}] =====")
    for cond, s in summary.items():
        print(f"  {cond[:4]:<6} proj {s['mean_proj_mm']:+.2f}±{s['mean_proj_se']:.2f}mm "
              f"(t={s['mean_proj_t']:+.1f})  占‖u‖ {s['mean_proj_mm']/max(s['median_u_mm'],1e-9):+.0%}  "
              f"proj>0 {100*s['proj_pos_rate']:>3.0f}%  朝供体 {100*s['toward_rate']:>3.0f}%  "
              f"n={s['n_pairs']}")

    suffix = args.dims if args.image == "ref" else f"{args.dims}_imgdon"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"probe_v4_{args.tag}_{suffix}.json")
    with open(out, "w") as f:
        import json
        json.dump(dict(tag=args.tag, dims=args.dims, image=args.image,
                       episodes=args.episodes, reps=args.reps,
                       summary=summary, rows=results), f,
                  ensure_ascii=False, indent=1)
    print(f"\n逐对明细：{out}")


if __name__ == "__main__":
    main()
