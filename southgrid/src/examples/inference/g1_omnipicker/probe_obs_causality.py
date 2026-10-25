"""同一策略、同一 state，只换**观测里的图像** —— 会把动作翻向吗？

要回答的问题（用户问题 #2 / #5）：
  "模型要做的是前往下一个工具的位置，既知道位置，也有距离，怎么就不能过去呢？"
  "这个视觉功能根本就没有发挥作用"

已知的一条硬线索（run_task3_tool_storage_scored.py 的注释里记着）：训练帧 0 的 state 与
演练 t=0 的 state 只差 4.7mm，但**训练集记录的 action Δy = -0.0631**（先往 -y 甩），
而演练里模型输出 **Δy = +0.0176**（反方向）。同一 state、相反动作 ⇒ 差异只可能来自
**观测**（图像 / prompt），不来自 state。

于是做这个对照：把两份观测分别喂给**同一个**策略服务，看输出。

  A. 训练观测：数据集 ep{ep} 帧{frame} 的 state + 从 mp4 解出来的 cam_head/cam_wrist_r
  B. 仿真观测：④ 演练 t=0 的 state(jsonl 首条) + attempt_start_*.png

判读：
  · A 复现训练记录的 action（Δy 为负、朝向 #0 扳手）而 B 不 → **策略没问题，是仿真
    开局观测不在分布内**。修的方向是"把开局摆成训练帧 0 的样子"，不是改模型。
  · A 和 B 输出接近 → 策略本身在训练观测上就已经偏右 → 问题在模型/训练，另找。

自校验：B 的输出应当复现演练里观察到的行为（④ 日志 `[cycle] t=0.0` 之后末端往哪走）。
若 B 复现不了，说明本脚本的观测构造有误，A/B 结论都不能用 —— 所以 B 是"对照组"。

用法：
  cd /home/dgl/lxc/SouthGrid
  /home/dgl/lxc/openpi/.venv/bin/python \
      src/examples/inference/g1_omnipicker/probe_obs_causality.py \
      --sim-jsonl /home/dgl/lxc/.runlogs/d22_goals_reversed.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

names = ["#0扳手", "#1螺丝刀", "#2电工刀(左)", "#3手电筒", "#4电工刀(右)"]
# 工具抓取点（base 系），用于判"策略想把末端送去哪件工具"
TOOL_REF = np.array([
    [0.5478929281234741, -0.08453216403722763, 0.16989122331142426],
    [0.5602749586105347, -0.2157687246799469, 0.18391813337802887],
    [0.5905364155769348, -0.2994392514228821, 0.1667836308479309],
    [0.5790082216262817, -0.4153374433517456, 0.17027008533477783],
    [0.5458616018295288, -0.49245551228523254, 0.16415002942085266],
], dtype=np.float64)


def nearest_tool(p) -> str:
    return names[int(np.argmin(np.linalg.norm(TOOL_REF - np.asarray(p), axis=1)))]


def decode_train_frame(dataset: str, episode: int, frame: int):
    """取训练帧图像。

    **不要用 lerobot 的 `decode_video_frames`**：本机上它对这批 HEVC 视频
    静默返回全零（第一版探针就是这么拿到 mean=0.0 的黑图，A 组整个作废）。
    `av` 直接解同一文件是好的（mean 88.23 / min 0 / max 255）。所以这里用 av。
    """
    import av
    out = {}
    for key in ("cam_head", "cam_wrist_r"):
        path = os.path.join(dataset, "videos/chunk-000",
                            f"observation.images.{key}", f"episode_{episode:06d}.mp4")
        c = av.open(path)
        arr = None
        for i, fr in enumerate(c.decode(video=0)):
            if i == frame:
                arr = fr.to_ndarray(format="rgb24")   # (H,W,3) uint8
                break
        c.close()
        if arr is None:
            raise RuntimeError(f"{path} 里没有第 {frame} 帧")
        if arr.mean() < 1.0:
            raise RuntimeError(f"{path} 第 {frame} 帧解码成全黑，解码器有问题")
        out[key] = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)), dtype=np.uint8)
    return out


def decode_many(dataset: str, key: str, episode: int, frames: list) -> dict:
    """一趟顺序解码，捞出多个帧号（按帧号随机 seek 是 O(n²)，这里 O(n)）。"""
    import av
    want = set(frames)
    path = os.path.join(dataset, "videos/chunk-000",
                        f"observation.images.{key}", f"episode_{episode:06d}.mp4")
    c = av.open(path)
    out = {}
    for i, fr in enumerate(c.decode(video=0)):
        if i in want:
            out[i] = np.ascontiguousarray(
                np.transpose(fr.to_ndarray(format="rgb24"), (2, 0, 1)), dtype=np.uint8)
            if len(out) == len(want):
                break
    c.close()
    return out


def load_sim_pngs(frames_dir: str):
    import cv2
    out = {}
    for key, fn in (("cam_head", "attempt_start_camera_head_color.png"),
                    ("cam_wrist_r", "attempt_start_camera_wrist_r_color.png")):
        p = os.path.join(frames_dir, fn)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # ④ 存盘时做过 RGB2BGR，这里逆回来
        out[key] = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.uint8)
    return out


def build_cases(args, st_tr, ac_tr, img_tr, st_sim, img_sim):
    """三组观测。B' 是关键：**state 换成训练的、只留仿真的图** —— 这样才真正
    只变一个变量。原来 A/B 同时变了 state 和图像（演练的 goal_y 是反向表的
    -0.4925，跟训练帧 0 的 -0.0845 根本不是一个数），结论没法归因。"""
    cases = [("A. 训练观测 (训练 state + 训练图)", st_tr, img_tr, ac_tr)]
    if args.isolate_image:
        # B'：训练 state + 仿真图。Δ 只能来自图像。
        cases.append(("B'. 只换图像 (训练 state + 仿真图)", st_tr, img_sim, ac_tr))
    cases.append(("C. 仿真观测 (仿真 state + 仿真图)", st_sim, img_sim, None))
    return cases


def report(tag: str, state: np.ndarray, images: dict, actions: np.ndarray,
           ref_action: np.ndarray | None = None) -> None:
    a0 = np.asarray(actions[0], dtype=np.float64)
    p_obs = np.asarray(state[7:10], dtype=np.float64)
    p_cmd = a0[7:10]
    d = p_cmd - p_obs
    print(f"\n===== {tag} =====")
    for k, v in images.items():
        print(f"  图 {k:<11} shape={v.shape} dtype={v.dtype} 均值={v.mean():6.1f}")
    print(f"  state  右末端 base = {np.round(p_obs, 4).tolist()}  "
          f"grip={state[16]:.3f}/{state[17]:.3f}  "
          f"goal_y={state[19]:+.4f} idx={state[21]:.0f}")
    print(f"  动作块 shape={np.shape(actions)}   首动作右末端 = {np.round(p_cmd, 4).tolist()}")
    print(f"  >>> 命令位移 Δ = {np.round(d, 4).tolist()}   |Δ|={np.linalg.norm(d)*1000:.1f}mm"
          f"   方向 Δy={d[1]:+.4f} ({'向 -y=向左/扳手侧' if d[1] < 0 else '向 +y=向右/电工刀(右)侧'})")
    print(f"  >>> 首动作落点最近工具 = {nearest_tool(p_cmd)}")
    print(f"  >>> 10 步后落点最近工具 = {nearest_tool(np.asarray(actions[min(9, len(actions)-1)])[7:10])}")
    if ref_action is not None:
        rp = np.asarray(ref_action[7:10], dtype=np.float64)
        print(f"  训练记录的同帧 action 右末端 = {np.round(rp, 4).tolist()}  "
              f"Δ={np.round(rp - p_obs, 4).tolist()}  "
              f"Δy={rp[1] - p_obs[1]:+.4f}")
    print(f"  首 3 个动作块的右末端 xyz：")
    for i in range(min(3, len(actions))):
        ai = np.asarray(actions[i], dtype=np.float64)
        print(f"    [{i}] {np.round(ai[7:10], 4).tolist()}  抓={ai[16]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--prompt", default="把所有工具从左到右放回工具箱")
    ap.add_argument("--sim-jsonl", default="/home/dgl/lxc/.runlogs/d22_goals_reversed.jsonl")
    ap.add_argument("--frames-dir", default="/home/dgl/lxc/.runlogs/task3_frames")
    ap.add_argument("--no-isolate", dest="isolate_image", action="store_false",
                    help="关掉 B' 组（默认开）")
    ap.set_defaults(isolate_image=True)
    ap.add_argument("--sweep-goal-y", action="store_true",
                    help="决定性实验：**同 state 同图，只把第 20 维 goal_y 换成 5 件工具各自的 y**，"
                         "看首动作方向变不变。变 → 模型确实读 goal_y（那 ④ 写错了值）；"
                         "不变 → goal_y 在帧 0 是死输入，模型靠视觉+记忆惯性走。")
    ap.add_argument("--replay", action="store_true",
                    help="**策略保真度**：把训练集自己的帧逐帧喂回模型，比对它输出的首动作和"
                         "训练记录的动作。误差一直小 → 策略在训练分布上是准的，问题在闭环走偏；"
                         "某段开始误差大 → 策略本身在那里就不准，别再怪起始位姿。")
    ap.add_argument("--replay-frames", type=str,
                    default="0,25,50,100,150,200,250,300,350,365,400,500,600,700,800,"
                            "900,1000,1200,1400,1600,1800,1965,2050",
                    help="要回放的训练帧号（逗号分隔）。视频与 state 是 1:1 同号。")
    args = ap.parse_args()

    import pandas as pd
    from openpi_client import websocket_client_policy
    from scipy.spatial.transform import Rotation as _R

    fs = sorted(glob.glob(args.dataset + "/data/**/*.parquet", recursive=True))
    df = pd.read_parquet(fs[args.episode])
    st_all = np.stack(df["observation.state"].values).astype(np.float64)
    ac_all = np.stack(df["action"].values).astype(np.float64)
    st_tr, ac_tr = st_all[args.frame], ac_all[args.frame]

    img_tr = decode_train_frame(args.dataset, args.episode, args.frame)
    print(f"[train] ep{args.episode} 帧{args.frame}  state 右末端 = "
          f"{np.round(st_tr[7:10], 4).tolist()}   idx={st_tr[21]:.0f} goal_y={st_tr[19]:+.4f}")

    # --replay 只用训练集，不碰仿真侧的任何输入（那个 jsonl 可以不传）。
    st_sim = img_sim = None
    if not args.replay:
        rec = json.loads(open(args.sim_jsonl, encoding="utf-8").readline())
        st_sim = np.asarray(rec["state"], dtype=np.float64).reshape(-1)
        img_sim = load_sim_pngs(args.frames_dir)
        print(f"[sim  ] {os.path.basename(args.sim_jsonl)} 首条 t={rec.get('t')}  "
              f"state 右末端 = {np.round(st_sim[7:10], 4).tolist()}   "
              f"idx={st_sim[21]:.0f} goal_y={st_sim[19]:+.4f}")
        dd = np.linalg.norm(st_tr[7:10] - st_sim[7:10]) * 1000
        print(f"[diff ] 两份 state 右末端位置相差 {dd:.1f} mm")

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[policy] metadata = {policy.get_server_metadata()}")

    got = {}
    if args.replay:
        fr_list = [int(s) for s in args.replay_frames.split(",") if s.strip()]
        fr_list = [f for f in fr_list if f < len(st_all)]
        print(f"\n===== 策略保真度：回放训练自己 ep{args.episode} 的 {len(fr_list)} 帧 =====")
        print("  帧号   t(s)  idx   训练记录首动作Δ      模型首动作Δ         位置误差  姿态误差")
        imgs = {k: decode_many(args.dataset, k, args.episode, fr_list)
                for k in ("cam_head", "cam_wrist_r")}
        errs = []
        for f in fr_list:
            if f not in imgs["cam_head"]:
                continue
            st_f = st_all[f]
            im = {k: imgs[k][f] for k in imgs}
            res = policy.infer({"state": st_f.astype(np.float32), "images": im,
                                "prompt": args.prompt})
            a = np.asarray(res["actions"])[0]
            a_ref = ac_all[f]
            dp = np.linalg.norm(a[7:10] - a_ref[7:10]) * 1000
            # 姿态误差：把四元数差换算成夹角
            dphi = np.degrees(np.linalg.norm(
                (_R.from_quat(a_ref[10:14]) * _R.from_quat(a[10:14]).inv()).as_rotvec()))
            errs.append((dp, dphi))
            print(f"  {f:5d} {f*0.05:6.2f}  {st_f[21]:.0f}   "
                  f"Δy={a_ref[7:10][1]-st_f[7:10][1]:+.4f}   "
                  f"Δy={a[7:10][1]-st_f[7:10][1]:+.4f}   "
                  f"{dp:8.1f}mm {dphi:8.1f}°")
        errs = np.array(errs)
        if len(errs):
            print(f"\n  位置误差：中位 {np.median(errs[:,0]):.1f}mm  最大 {errs[:,0].max():.1f}mm")
            print(f"  姿态误差：中位 {np.median(errs[:,1]):.1f}°   最大 {errs[:,1].max():.1f}°")
            print(f"  → {'策略在训练分布上是准的（误差小）→ 问题在闭环走偏，不是策略不会做' if np.median(errs[:,0]) < 40 else '策略在训练帧上就偏得厉害 → 别只怪起始位姿'}")
        return
    if args.sweep_goal_y:
        # 只动 state[19]（goal_y），图和其余 21 维一字不改。
        print("\n===== goal_y 扫描：同 state 同图，只换第 20 维 =====")
        print(f"  （图的 y 顺序 由大到小 = 桌面从左到右；grip 保持张开 {st_tr[16]:+.3f}）")
        dys = []
        for k in range(5):
            st_c = st_tr.copy()
            st_c[19] = TOOL_REF[k][1]     # goal_y ← 第 k 件工具的 y
            st_c[21] = 1.0                # cycle_idx 固定在 1（=「去抓最左那件」）
            res = policy.infer({"state": st_c.astype(np.float32), "images": img_tr,
                                "prompt": args.prompt})
            a0 = np.asarray(res["actions"])[0][7:10]
            dy = a0[1] - st_c[7:10][1]
            dys.append(dy)
            same_dir = np.sign(a0[1] - st_c[7:10][1]) == np.sign(st_c[19] - st_c[7:10][1])
            print(f"  goal_y={st_c[19]:+.4f} ({names[k]:<11s}) → 首动作 y={a0[1]:+.4f} "
                  f"Δy={dy:+.4f}  指向 {nearest_tool(a0):<11s} "
                  f"{'(与 goal_y 同向 ✓)' if same_dir else '(与 goal_y 反向 ✗)'}")
        dys = np.array(dys)
        print(f"\n  五组 Δy 极差 = {(dys.max()-dys.min())*1000:.1f}mm   "
              f"标准差 = {dys.std()*1000:.1f}mm   （全都同号? "
              f"{'是 → goal_y 没改变方向，是死输入' if len(set(np.sign(dys))) == 1 else '否 → goal_y 有作用'})")
        return
    for tag, st_c, img_c, ac_c in build_cases(args, st_tr, ac_tr, img_tr, st_sim, img_sim):
        res = policy.infer({"state": st_c.astype(np.float32), "images": img_c,
                            "prompt": args.prompt})
        acts = np.asarray(res["actions"])
        got[tag] = acts
        report(tag, st_c, img_c, acts, ref_action=ac_c)

    print(f"\n===== 结论 =====")
    keys = list(got)
    for tag in keys:
        a0 = got[tag][0][7:10]
        st_c = next(c[1] for c in build_cases(args, st_tr, ac_tr, img_tr, st_sim, img_sim)
                    if c[0] == tag)
        print(f"  {tag:<34} 首动作 Δy = {(a0[1]-st_c[7:10][1]):+.4f} → 指向 {nearest_tool(a0)}")
    # 训练记录的"标准答案"：帧 0 命令 Δy 在 100 集里全是负的（-0.0581…-0.0631）
    ref_dy = ac_tr[7:10][1] - st_tr[7:10][1]
    print(f"  训练记录该帧的 Δy                      = {ref_dy:+.4f} "
          f"（100 集里帧 0 的 Δy 恒为负，mean -0.0616）")
    a_dy = got[keys[0]][0][7:10][1] - st_tr[7:10][1]
    print(f"\n  A 复现训练答案? {'是' if np.sign(a_dy) == np.sign(ref_dy) else '否'}")
    if args.isolate_image and len(keys) >= 3:
        b_dy = got[keys[1]][0][7:10][1] - st_tr[7:10][1]
        print(f"  B' 只用训练 state、换成仿真的图 → Δy = {b_dy:+.4f}  "
              f"({'仍是负 → 图像没改变方向' if np.sign(b_dy) == np.sign(ref_dy) else '翻成正 → **图像是罪魁**'})")
        print(f"  → 结论：{'差异来自**图像**（仿真开局观测不在训练分布内）' if np.sign(b_dy) != np.sign(ref_dy) else '差异**不**来自图像，回头查 state/prompt'}")


if __name__ == "__main__":
    main()
