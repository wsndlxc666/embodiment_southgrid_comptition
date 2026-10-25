#!/usr/bin/env python3
"""读 --dump_chunks 的 JSONL，把策略"每一步想抓哪个"读出来。

原理：策略输出的动作块 chunk[:, 7:10] 是**右臂末端在 base 系下的目标位置**，
和 tool_refs_d19goal.json 里那 6 个参照点(5 工具抓取点 + 1 盒子落点)**同坐标系同单位**。
所以把一次推理的 chunk 末步位置和 6 个点比 dist、取 argmin，就是这个 chunk
"打算去哪"的读数。

**动作块有多少真被下发，取决于执行模式**(转储里的 exec_mode 字段)：
  * `chunk`：整块 20 步依次下发(每条动作占 10 个 env 步 = 0.05s = 训练 20fps)，
    所以 **chunk 末步 = 这次决策实际执行到的终点**，"瞄哪件"是真读数；块末的闭爪指令
    也会真的执行。
  * `first`：只下发第 0 步，**其余 19 步全部丢弃**。这时末步只是个"被丢掉的计划终点"，
    块末的闭爪永远不会被执行 —— 本脚本会照实标出来，别把 first 模式的瞄准读数当成
    "机器人去了哪"。

用法:
    python analyze_chunks.py <dump.jsonl> [--tool_refs tool_refs_d19goal.json]
                             [--goal-names 扳手,螺丝刀,电工刀(左),手电筒,电工刀(右),盒子]

两种转储格式都吃：
  * run_task3_tool_storage_scored.py 的：一个工具一段 prompt，靠 prompt 反推下标->工具名。
  * g1_omnipicker_policy_clear_eval.py 的：全程一条恒定 prompt，但每条记录了 5 把工具的
    世界坐标(tools_world)和名字(tool_names)，直接一一匹配即可，不依赖 prompt。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))

# 顺序 = conf 里 _TOOL_BODY_JOINT_NAMES / CLEAR_ORDER 的顺序 = base 系 -Y 递增(从左到右)。
# 这两个名字以前写反过(#1 是螺丝刀、#2 是电工刀(左))，是拿运行数据的坐标一一对上才定的。
GOAL_NAMES_DEFAULT = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)", "盒子"]


def load_dump(path: str) -> list[dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tool_refs", default=os.path.join(base_dir, "tool_refs_d19goal.json"))
    ap.add_argument("--csv", default="", help="把逐次推理的时间线写成 CSV 便于画图")
    ap.add_argument("--goal-names", default="",
                    help="逗号分隔的参照点名字，覆盖自动反推。顺序须与 tool_refs json 里的 goals 一致，"
                         "默认 扳手,螺丝刀,电工刀(左),手电筒,电工刀(右),盒子")
    ap.add_argument("--no-timeline", action="store_true",
                    help="不逐行打印时间线（600 次推理时太长，只看汇总）")
    args = ap.parse_args()

    with open(args.tool_refs, "r", encoding="utf-8") as f:
        tr = json.load(f)
    goals = np.asarray(tr.get("goals") or tr["tool_refs"], dtype=np.float64)
    names = list(GOAL_NAMES_DEFAULT[: len(goals)])
    print(f"参照点 {len(goals)} 个 (来自 {os.path.basename(args.tool_refs)})")

    recs = load_dump(args.dump)
    if not recs:
        print("dump 为空"); return

    # base_world 缺失时补算：由构造 state[7:10] == ee_world - base_world，
    # 所以 base_world = ee_world - state[7:10]。补出来才能把工具位置换算到 base 系。
    n_patch = 0
    for r in recs:
        if not r.get("base_world") and r.get("ee_world") and r.get("state") and len(r["state"]) >= 10:
            r["base_world"] = list(np.asarray(r["ee_world"], float) - np.asarray(r["state"][:10], float)[7:10])
            n_patch += 1
    if n_patch:
        print(f"(base_world 缺失，已由 ee_world - state[7:10] 补算 {n_patch} 条)")

    trials = list(dict.fromkeys(r.get("trial") for r in recs if r.get("trial") is not None))
    scope = f"trial {'/'.join(map(str, trials))}" if trials else f"阶段 {', '.join(dict.fromkeys(r['prompt'] for r in recs))}"
    print(f"推理次数 {len(recs)}  ({scope})")

    # ---- 执行模式：决定 chunk 里有多少步真被下发 -----------------------------
    # 老转储没有 exec_mode 字段(2026-09-14 之前)，靠相邻记录 step 的增量反推：
    # first 模式每次推理只推进 query_every(默认10)个 env 步，chunk 模式推进
    # action_repeat×horizon(默认 10×20=200)个。两者差 20 倍，不会看错。
    default_mode = "chunk"      # runner 的转储没这个字段，但它本来就是整块下发
    if all("exec_mode" not in r for r in recs) and len(recs) >= 3:
        dsteps = [b["step"] - a["step"] for a, b in zip(recs, recs[1:])
                  if isinstance(a.get("step"), int) and isinstance(b.get("step"), int)]
        dsteps = [d for d in dsteps if d > 0]
        if dsteps:
            med = float(np.median(dsteps))
            default_mode = "chunk" if med >= 50 else "first"
            print(f"(转储无 exec_mode，按相邻推理的 step 增量中位 {med:.0f} 推断为 "
                  f"{default_mode} 模式)")
    modes = {r.get("exec_mode", default_mode) for r in recs}
    ar = {r.get("action_repeat") for r in recs if r.get("action_repeat") not in (None, 1)}
    if modes == {"first"}:
        print("执行模式 first：**只下发第 0 步，其余 19 步丢弃**。"
              "下面的'瞄准/块末'读数都是被丢掉的计划终点，不代表机器人去了哪。")
    else:
        note = f"，每条动作占 {sorted(ar)[0]} 个 env 步" if ar else ""
        print(f"执行模式 {sorted(modes)[0]}：整块 20 步依次下发{note}"
              f"（每次推理覆盖约 1.0s 仿真时间，与训练 20fps 同速）")
    if len(modes) > 1:
        print(f"  ⚠ 同一份转储里混了多种 exec_mode {sorted(modes)}，逐条按各自的模式算。")

    # 名字来源优先级：命令行 --goal-names > 转储里自带 goal_names > 下面的自动反推
    if args.goal_names:
        names = [x.strip() for x in args.goal_names.split(",")][: len(goals)]
    elif recs[0].get("goal_names"):
        names = list(recs[0]["goal_names"])[: len(goals)]

    # ---- 0) 坐标系自检：state[7:10] 应当 ≈ ee_world - base_world -------------
    # 注意 state[7:10] 是**真值**(由 obs 直接给)，ee_world-base_world 是我们拿两个
    # 世界坐标自己减的。两者差在哪，就说明我们对 base 系的理解差在哪。
    print("\n=== 坐标系自检 ===")
    devs, bases = [], []
    for r in recs:
        st, ee, bs = r.get("state"), r.get("ee_world"), r.get("base_world")
        if st and ee and bs and len(st) >= 10:
            devs.append(np.asarray(st[7:10], float) - (np.asarray(ee, float) - np.asarray(bs, float)))
            bases.append(np.asarray(bs, float))
    if devs:
        D = np.asarray(devs)
        B = np.asarray(bases)
        med = np.median(D, axis=0)
        mx = np.abs(D).max() * 1000
        print(f"|state[7:10] - (ee_world-base_world)|  最大={mx:.1f} mm  "
              f"中位={np.median(np.abs(D).max(axis=1))*1000:.1f} mm")
        print(f"  分轴中位偏差: dx={med[0]*1000:+.1f}  dy={med[1]*1000:+.1f}  dz={med[2]*1000:+.1f} mm")
        print(f"  base_world 自身抖动: " + "  ".join(
            f"d{a}={B[:, i].std()*1000:.1f}mm" for i, a in enumerate("xyz")))
        if mx < 5.0:
            print("  ✓ ≈0：state 的右臂位姿确实是 base 系，和参照点同一套坐标")
            yaw = None
        else:
            # 底座绕 z 偏航 θ 时，真值(base 系)与"世界系直减"差一个旋转：
            #   dx ≈ -y·θ,  dy ≈ +x·θ,  dz ≈ 0
            # 判别靠**形状**(偏差是否只在 y、是否与位置成比例)，不靠绝对大小 ——
            # 偏差多大取决于这一轮顶得多狠，用固定阈值会时对时错。
            st07 = np.array([r["state"][7:10] for r in recs if r.get("state") and len(r["state"]) >= 10])
            x_h, y_h = float(np.median(st07[:, 0])), float(np.median(st07[:, 1]))
            yaw = med[1] / x_h if abs(x_h) > 0.05 else float("nan")
            pred_x = -y_h * yaw
            yaw_deg = np.degrees(yaw)
            # y 主导 + x 能被 -y·θ 解释 + z≈0 + θ 在合理量级
            yaw_ok = (np.isfinite(yaw_deg) and 0.05 < abs(yaw_deg) < 20.0
                      and abs(med[1]) > 2.0 * abs(med[0])
                      and abs(med[1]) > 2.0 * abs(med[2])
                      and abs(med[0] - pred_x) * 1000 < max(2.0, 0.35 * abs(med[1]) * 1000))
            if yaw_ok:
                print(f"  ✓ 是**底座被反作用力顶得偏航**，不是坐标系搞错了："
                      f"偏差几乎全在 y、x/z≈0，且与手臂水平位置成比例")
                print(f"    反解偏航角 ≈ {yaw_deg:+.2f}°（手臂 x≈{x_h:+.3f} m 处 dy={med[1]*1000:+.1f} mm）"
                      f"；预测 dx={pred_x*1000:+.1f} mm，实测 {med[0]*1000:+.1f} mm，对得上")
                b_jit = float(B[:, :2].std(axis=0).max())   # 逐轴取，别把两轴混在一起算
                if b_jit > 0.002:
                    print(f"    base_world 自身在动(逐轴抖动最大 {b_jit*1000:.1f} mm)，"
                          f"正是被顶着的直接证据")
                print(f"    量级 {mx:.1f} mm 取决于这一轮顶得多狠；这只是记录用的世界坐标有偏，"
                      f"**下面的瞄准读数不受影响**(动作块和参照点常量都在 base 系，state[7:10] 是真值)。")
            else:
                print("  ⚠ 偏差远超数值误差，且不是底座偏航的特征(偏差不在 y、或与位置不成比例)。")
                print("    要么 state 不是 base 系，要么 ee_world/base_world 记的是另一个 site/body"
                      " —— 那下面的瞄准读数就全是错的，先查这个再往下看。")
    else:
        print("  缺 ee_world/base_world，跳过")

    # ---- 1) 反推 参照点下标 -> 工具名 ---------------------------------------
    # 每个阶段跑哪件工具是已知的(runner 的 ph["site"]/prompt)；把该阶段工具在 base 系
    # 的位置和参照点比，最近的那个就是"这件工具自己的参照点"。
    # 最后一个参照点是盒子(goals[-1])，**不参与**"工具<->工具"匹配——否则它会把
    # 某件工具抢走，导致两件工具撞到同一个下标、另一件变成"未识别"。
    n_tool_goals = len(goals) - 1          # 末尾是盒子
    names[-1] = "盒子"
    names = names[: len(goals)]

    # ---- 名字从哪来 ---------------------------------------------------------
    # **默认名字是按构造定的，不靠反推**：build_d19_goal_dataset.py 里
    #   TOOL_REF[i] = 各集第 i 次抓取时刻的 ee 位置(时间顺序)
    # 而采集器 CLEAR_ORDER = [0,1,2,3,4] = TOOL_NAMES 顺序 = base 系 -Y 递增(从左到右)，
    # 抓取顺序和时间顺序一致，所以 tool_refs[i] 就是 TOOL_NAMES[i]。
    # 曾经这里用"和运行中工具位置做贪心最近匹配"来反推，那是**不可靠的**：
    # 每次种植的布局有抖动，参照点间距只有 8~13cm，而抖动可达 10cm 以上，
    # 贪心会把相邻两件工具对调。所以下面的反推降级成"交叉核对"，不再决定名字。
    defaulted = not (args.goal_names or recs[0].get("goal_names"))
    if defaulted:
        print("\n=== 参照点下标 -> 工具名（按构造顺序，见 build_d19_goal_dataset.py） ===")
        for j in range(len(goals)):
            print(f"  #{j} {names[j]}  {np.round(goals[j], 4).tolist()}")

    # ---- 交叉核对：运行数据里各工具的实际位置，是否能对上同名参照点 ----------
    if n_tool_goals > 0:
        tool_pos: dict[str, np.ndarray] = {}
        if recs[0].get("tools_world") and recs[0].get("tool_names"):
            tn = list(recs[0]["tool_names"])
            for k, tname in enumerate(tn):
                pts = [np.asarray(r["tools_world"][k]) - np.asarray(r["base_world"])
                       for r in recs
                       if r.get("tools_world") and len(r["tools_world"]) > k
                       and r["tools_world"][k] is not None and r.get("base_world")]
                if pts:
                    tool_pos[tname] = np.mean(pts, axis=0)
        else:
            for prompt in dict.fromkeys(r["prompt"] for r in recs):
                pts = [np.asarray(r["tool_world"]) - np.asarray(r["base_world"])
                       for r in recs
                       if r["prompt"] == prompt and r.get("tool_world") and r.get("base_world")]
                if pts:
                    tool_pos[prompt] = np.mean(pts, axis=0)

        if len(tool_pos) >= 2:
            print("\n=== 交叉核对：工具实际位置 vs 同名参照点 ===")
            bad = []
            for label, p in tool_pos.items():
                if label not in names[:n_tool_goals]:
                    continue
                j = names.index(label)
                d_same = float(np.linalg.norm(goals[j] - p))
                # 该工具离它"本该是"的那个点，比离任何一个别的点都近吗
                d_all = [float(np.linalg.norm(goals[k] - p)) for k in range(n_tool_goals)]
                j_near = int(np.argmin(d_all))
                flag = "" if j_near == j else f"  ⚠ 最近的是 #{j_near}({names[j_near]})"
                if j_near != j:
                    bad.append(label)
                print(f"  {label:<11} 距 #{j}(同名) {d_same*100:5.1f}cm  "
                      f"最近 #{j_near} {min(d_all)*100:5.1f}cm{flag}")
            if bad:
                print(f"  ⚠ {len(bad)} 件工具的实际位置离别的参照点更近（{', '.join(bad)}）。"
                      f"\n     这多半是种植抖动(参照点间距只有 8~13cm)造成的，不代表名字错；"
                      f"\n     名字以构造顺序为准。要覆盖可用 --goal-names。")
            else:
                print("  ✓ 每件工具的实际位置都是离同名参照点最近，命名与运行数据自洽")

    # ---- 2) 逐次推理读"瞄哪个" ---------------------------------------------
    rows = []
    for i, r in enumerate(recs):
        chunk = np.asarray(r["chunk"], dtype=np.float64)  # (horizon, 18)
        if chunk.ndim != 2 or chunk.shape[1] < 18:
            continue
        ee_track = chunk[:, 7:10]                      # 规划的未来末端轨迹(base 系)
        d_end = np.linalg.norm(goals - ee_track[-1], axis=1)
        j_end = int(np.argmin(d_end))
        # 整条规划轨迹对每个参照点的最近距离 -> 这次规划"顺路"经过谁
        d_track = np.linalg.norm(goals[:, None, :] - ee_track[None, :, :], axis=2).min(axis=1)
        j_track = int(np.argmin(d_track))
        # 闭爪(=norm>0.5，norm 1.0=合爪)在块内第一次出现在第几步；None = 整块都张着
        closed = np.where(chunk[:, 16] > 0.5)[0]
        first_close = int(closed[0]) if len(closed) else None
        # 实际会被下发的步数：first 模式只有第 0 步，chunk 模式是整块
        n_exec = 1 if r.get("exec_mode", default_mode) == "first" else len(chunk)
        rows.append(dict(
            i=i, t=r["t"], phase_t=r["phase_t"], trial=r.get("trial"),
            prompt=r["prompt"], step=r["step"],
            aim=j_end, aim_name=names[j_end], d_end=float(d_end[j_end]),
            track_aim=j_track, track_name=names[j_track], d_track=float(d_track[j_track]),
            grip_start=float(chunk[0, 16]), grip_end=float(chunk[-1, 16]),
            n_exec=int(n_exec),
            # 第 0 步的闭爪指令（唯一在 first 模式下真正执行的）
            grip_exec0=float(chunk[0, 16]),
            # **实际下发范围内**出现过闭爪指令吗 —— 这才是"有没有下过合爪命令"的真相
            grip_exec_close=bool((chunk[:n_exec, 16] > 0.5).any()),
            first_close_step=first_close,
            # 规划轨迹的位移幅度：这次规划想让末端走多远
            travel=float(np.linalg.norm(ee_track[-1] - ee_track[0])),
            ee0=[round(float(x), 4) for x in ee_track[0]],
            eeN=[round(float(x), 4) for x in ee_track[-1]],
        ))

    if args.no_timeline:
        print("\n=== 时间线（--no-timeline，略） ===")
    else:
        print("\n=== 时间线（每次推理一行；aim=末步位置最近的参照点） ===")
        cur_trial = object()
        for r in rows:
            if r["trial"] != cur_trial:                 # 换 trial 时插一行分隔
                cur_trial = r["trial"]
                if r["trial"] is not None:
                    print(f"  --- trial {r['trial']} ---")
            _cl = "" if r["first_close_step"] is None else f" 闭@{r['first_close_step']:02d}"
            print(f"  t={r['t']:6.1f}s ({r['phase_t']:5.1f}s) {r['prompt'][:11]:<11} "
                  f"瞄准={r['aim_name']:<10} 距={r['d_end']*100:5.1f}cm  "
                  f"夹爪 {r['grip_start']:.2f}->{r['grip_end']:.2f}{_cl}  "
                  f"本次规划位移={r['travel']*100:5.1f}cm")

    # ---- 2.5) 闭爪指令：规划在第几步 / 实际有没有被下发 ---------------------
    # 这一段是判断"这个台架有没有真的下令合爪"的唯一依据。first 模式下块末被丢弃，
    # 只看 grip_end 会得到"它从不闭爪"的错误结论。
    print("\n=== 闭爪指令（右爪 norm>0.5 = 合爪；约定 norm 1.0=合、0.048=张）===")
    n = len(rows)
    step0_close = sum(1 for r in rows if r["grip_exec0"] > 0.5)
    n_exec_close = sum(1 for r in rows if r["grip_exec_close"])
    print(f"  第 0 步就是闭爪: {step0_close}/{n}")
    print(f"  **实际下发范围内**含闭爪指令: {n_exec_close}/{n} = {n_exec_close/n*100:.1f}%")
    fc = [r["first_close_step"] for r in rows if r["first_close_step"] is not None]
    if not fc:
        print("  整块 20 步里从来没出现过闭爪指令 —— 它压根没规划过抓取")
    else:
        print(f"  块内首次闭爪在第几步: 中位 {int(np.median(fc))}, 范围 {min(fc)}~{max(fc)}"
              f"（{len(fc)}/{n} 块规划过抓取）")
        from collections import Counter as _C
        c2 = _C(fc)
        print("  分布(步:块数): " + "  ".join(f"{k}:{c2[k]}" for k in sorted(c2)))
    if n_exec_close == 0 and fc:
        print("  ⚠ 所有闭爪都落在被丢弃的那部分里 —— 模型规划了抓取，但这个台架一次都没发下去。")

    # ---- 3) "抓取尝试" = 连续若干次瞄准同一个参照点 -------------------------
    print("\n=== 抓取尝试分段（连续瞄准同一参照点，且规划位移不接近 0） ===")
    MIN_RUN = 3          # 至少连续 3 次推理
    MIN_TRAVEL = 0.01    # 至少规划要动 1cm，否则是"原地发呆"不算尝试
    attempts = []
    cur = None
    for r in rows:
        moving = r["travel"] >= MIN_TRAVEL
        if cur and cur["aim"] == r["aim"] and moving:
            cur["n"] += 1; cur["t1"] = r["t"]; cur["d_min"] = min(cur["d_min"], r["d_end"])
        else:
            if cur and cur["n"] >= MIN_RUN:
                attempts.append(cur)
            cur = dict(aim=r["aim"], name=r["aim_name"], prompt=r["prompt"],
                       t0=r["t"], t1=r["t"], n=1, d_min=r["d_end"]) if moving else None
    if cur and cur["n"] >= MIN_RUN:
        attempts.append(cur)

    if not attempts:
        print("  没有识别到成形的抓取尝试")
    for k, a in enumerate(attempts, 1):
        print(f"  尝试{k}: 瞄准 {a['name']:<10} ({a['t0']:.0f}s~{a['t1']:.0f}s, {a['n']}次推理) "
              f"最近规划距 {a['d_min']*100:.1f}cm   [阶段 {a['prompt']}]")
    if len(attempts) > 1:
        same = len({a["aim"] for a in attempts}) == 1
        print(f"\n  → 共 {len(attempts)} 次尝试，瞄准 {len({a['aim'] for a in attempts})} 个不同参照点"
              + ("（**全部指向同一个**）" if same else ""))

    # ---- 4) 各参照点被瞄准的次数 -------------------------------------------
    print("\n=== 各参照点被瞄准的推理次数占比 ===")
    from collections import Counter
    c = Counter(r["aim_name"] for r in rows)
    for nm, n in c.most_common():
        print(f"  {nm:<12} {n:>4} 次  {n/len(rows)*100:5.1f}%")

    # ---- 5) 交叉表：prompt 说抓谁  vs  实际瞄谁 ------------------------------
    # 这张表是"它在不在听指令"的直接证据。对角线 = 瞄对了。
    # eval 转储全程一条恒定 prompt(不是工具名)，那时按 trial 分组，只看瞄准分布。
    by_prompt = all(r["prompt"] in names for r in rows) and \
        len({r["prompt"] for r in rows}) > 1
    key = (lambda r: str(r["prompt"])) if by_prompt else \
        (lambda r: f"trial{r['trial']}" if r["trial"] is not None else "全部")
    rows_key = "prompt(行)" if by_prompt else ("trial(行)" if rows[0]["trial"] is not None else "(行)")
    print(f"\n=== {rows_key} × 实际瞄准(列) 推理次数 ===")
    phases = list(dict.fromkeys(key(r) for r in rows))
    cols = [names[j] for j in range(len(goals))]
    w = max(11, *(len(x) for x in cols))
    print(f"  {'':<11}" + "".join(f"{x:>{w+1}}" for x in cols)
          + (f"{'瞄对%':>8}" if by_prompt else ""))
    total_correct = total_all = 0
    for p in phases:
        sub = [r for r in rows if key(r) == p]
        cnt = Counter(r["aim_name"] for r in sub)
        line = f"  {p[:11]:<11}" + "".join(f"{cnt.get(x,0):>{w+1}}" for x in cols)
        if by_prompt:
            hit = cnt.get(p, 0)
            total_correct += hit; total_all += len(sub)
            line += f"{hit/len(sub)*100:>7.1f}%"
        print(line)
    if by_prompt:
        print(f"\n  → 全程「瞄对了 prompt 指定的那件工具」的比例: "
              f"{total_correct}/{total_all} = {total_correct/total_all*100:.1f}%")
        print("     (随机瞎瞄 5 件工具也有 ~20%，所以接近 20% 就等于没在听指令)")
    else:
        print(f"\n  → 全程用的是一条恒定指令「{rows[0]['prompt']}」，这张表只反映"
              f"「同一句指令下它实际去了哪几件」，不构成听不听指令的证据。")

    # ---- 6) 各工具被"深度接近"的情况 ---------------------------------------
    print("\n=== 深度接近（规划末步距该参照点 < 5cm）===")
    deep = [r for r in rows if r["d_end"] < 0.05]
    if not deep:
        print("  全程没有任何一次规划真正到达过 5cm 以内")
    else:
        dc = Counter(r["aim_name"] for r in deep)
        for nm, n in dc.most_common():
            ds = [r["d_end"] for r in deep if r["aim_name"] == nm]
            phs = sorted({r["prompt"] for r in deep if r["aim_name"] == nm})
            print(f"  {nm:<12} {n:>3} 次   最近 {min(ds)*100:4.1f}cm   "
                  f"出现在这些 prompt 下: {', '.join(phs)}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n时间线已写 {args.csv}")


if __name__ == "__main__":
    main()
