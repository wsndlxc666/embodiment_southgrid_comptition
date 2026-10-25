#!/bin/bash
# idxS 训练：**接 idxR/11999 往下推，只换数据源**（recipe 逐字不变 ⇒ 单变量可归因）。
#
# 数据 = g1_omnipicker_clear5_d22idxS_100ep = 把每集的 seg1..seg4 按 24 种排列轮换重排。
# 为什么：原数据 100/100 集 cycle_idx 跳变帧完全相同 ⇒ idx 是帧号纯函数 ⇒
# 模型学成"按视觉进度走时间表"而不是"读相位"。详见 config.py 里那段注释。
#
# ⚠ 本轮第一手修法在 harness（--progress_gate 默认已改回 grip），**不需要等训练**。
#   早上先用现有 idxR/11999 + 修好的 harness 跑闭环；本档是后备。
#
# 用法: train_d22idxS.sh smoke        # 20 步，全链路冒烟
#       train_d22idxS.sh              # 正式，默认 12000 步（≈5.3h @1.6s/step）
#       train_d22idxS.sh 6000         # 指定步数
set -euo pipefail

MODE="${1:-12000}"
CONFIG="pi05_g1_omnipicker_clear5_d22idxS_lora_12k"
EXP="clear5_d22idxS_lora_100ep"
LOG="/home/dgl/lxc/.runlogs/train_d22idxS.log"
PY=/home/dgl/lxc/openpi/.venv/bin/python
DATA=/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep
cd /home/dgl/lxc/openpi
export HF_LEROBOT_HOME=/home/dgl/lxc/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------- 显存
# step-0 OOM 的真因是**碎片**不是显存不够；正解是预分配一整个大块。
# 见 memory openpi-train-oom-is-fragmentation。
# 代价：22.6GB 被训练独占，**训练期间起不了 OrcaLab 仿真** —— 别在训的时候开仿真。
if [ "${FORCE_PLATFORM_ALLOC:-0}" = "1" ]; then
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    ALLOC_NOTE="platform（慢，稳态 ~2.6 s/step；峰值低，可与仿真共存）"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
    ALLOC_NOTE="BFC 预分配整块 0.92（快，~1.6 s/step；训练期间仿真起不来）"
fi
mkdir -p /home/dgl/lxc/.runlogs

# ---------------------------------------------------------------- 起训前硬闸
echo "===== 起训前检查 $(date '+%F %T') ====="
for f in src/openpi/training/cycle_index.py src/openpi/training/config.py \
         src/openpi/training/data_loader.py; do
    if ! "$PY" -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" 2>&1; then
        echo "✗ 语法检查没过：$f —— 先修好再起训" >&2
        exit 1
    fi
done
echo "✓ 三个被改过的文件语法都 OK"

"$PY" - <<'PYEOF'
import itertools, json, pathlib, sys
import numpy as np, pandas as pd
import openpi.training.config as C

errs = []
DATA = pathlib.Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep")
cfg = C.get_config("pi05_g1_omnipicker_clear5_d22idxS_lora_12k")
d = cfg.data.base_config
w = cfg.model.loss_weights
m = sum(w) / len(w)
wl = pathlib.Path(cfg.weight_loader.params_path)
WANT_PHASE = ((0, 3.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0))
WANT_LW = [0.05] * 7 + [1.0] * 9 + [8.0] * 2 + [5.0] * 1 + [3.0] * 3 + [1.0] * 10

print(f"✓ config: {cfg.name}  exp={cfg.exp_name}  steps={cfg.num_train_steps}")
print(f"✓ repo_id: {cfg.data.repo_id}")

# norm_stats 必须真的加载到：`asset_id` 默认取 repo_id，新 repo_id 找不到就静默 None，
# 然后 data_loader 才抛 "Normalization stats not found"。在起训前就拦住。
# ⚠ 必须看 **create() 的产物**，不能看 `cfg.data.base_config` ——
#   norm_stats 是 `create_base_config()` 用 dataclasses.replace 挂到一个**新对象**上的，
#   而 base_config 是**传进去的输入**，它的 norm_stats 永远是 None。查错对象会得到
#   一个恒定报错的假闸（我第一版就是这么写的）。
_bc = cfg.data.create(cfg.assets_dirs, cfg.model)
_ns = _bc.norm_stats
if _ns is None:
    errs.append("norm_stats 为 None —— 起训必抛 ValueError；查 assets.asset_id / assets_dir")
    print("✗ norm_stats: None")
else:
    print(f"✓ norm_stats: 已加载 {len(_ns['state'].mean)} 维"
          f"  右爪 state.mean[16:18]={[round(float(x), 4) for x in _ns['state'].mean[16:18]]}"
          f"  idx(dim21) q01/q99={float(_ns['state'].q01[21]):.1f}/{float(_ns['state'].q99[21]):.1f}")
print(f"✓ 相位权重: {d.cycle_phase_weights}")
print(f"✓ 边界过采样: window={d.cycle_boundary_window} weight={d.cycle_boundary_weight}"
      f"  反事实 prob={d.cycle_counterfactual_prob} images={d.cycle_counterfactual_images}")
print(f"✓ 右爪16/17 = {[round(x/m,3) for x in w[16:18]]}（归一化后）"
      f"  左爪14/15 = {[round(x/m,3) for x in w[14:16]]}"
      f"  索引19/20/21 = {[round(x/m,3) for x in w[19:22]]}")

if tuple(d.cycle_phase_weights) != WANT_PHASE:
    errs.append(f"相位权重不是设计的 {WANT_PHASE}")
if list(w) != WANT_LW:
    errs.append(f"loss_weights 不是设计的（长度 {len(w)}，右爪 {w[16:18]}）")
if cfg.data.repo_id != "g1_omnipicker_clear5_d22idxS_100ep":
    errs.append(f"repo_id 不是 idxS 数据集: {cfg.data.repo_id}")
if not wl.is_dir():
    errs.append(f"热启动源不存在: {wl}")
if "idxR" not in str(wl) or "11999" not in str(wl):
    errs.append(f"热启动源不是 idxR/11999: {wl}")

# ---- 数据集硬闸：必须真的是重排过的，否则这一轮白跑 ----
print(f"\n--- 数据集核对 {DATA.name} ---")
if not DATA.is_dir():
    errs.append(f"数据集不存在: {DATA}（先跑 build_phase_shuffled_dataset.py）")
else:
    meta = json.loads((DATA / "meta/info.json").read_text())
    n_ep = sorted(int(p.stem.split("_")[-1])
                  for p in (DATA / "data/chunk-000").glob("episode_*.parquet"))
    print(f"  集数 {meta['total_episodes']}（磁盘上 {len(n_ep)}）  总帧 {meta['total_frames']}"
          f"  fps {meta['fps']}")
    if len(n_ep) != 100:
        errs.append(f"数据集不完整：磁盘 {len(n_ep)} 集，应为 100")
    if meta["total_frames"] != 210100:
        errs.append(f"总帧数 {meta['total_frames']} != 210100")

    # 抽 6 集核对重排。⚠ **不能用 dim21 的"段长签名"当判据** ——
    # seg1..seg4 恒为 400 帧，任何排列的段长序列都是
    # [(1,366),(2,400),(3,400),(4,400),(5,400),(6,135)]，对排列**完全不变**，
    # 用它做判据永远"看不出差别"（与 verify 脚本里那个 BOUND[k] 下标错误同一类陷阱）。
    # 真正的判据是**反推排列**：从 slot 的 dim19 中位数反查它放的是源里哪一段，
    # 得到的排列必须恰好等于 PERMS[ep%24]。
    SRC = pathlib.Path("/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22_100ep")
    PERMS = list(itertools.permutations(range(1, 5)))
    BOUND = [0, 366, 766, 1166, 1566, 1966, 2101]   # 实测 100/100 集相同

    def arr(df, c):
        x = df[c].to_numpy()
        if x.dtype == object:
            x = np.stack([np.asarray(v, float) for v in x])
        return np.asarray(x, float)

    # 源里每一段的 goal_y（工具 y，重排不改变它）
    _s = pd.read_parquet(SRC / "data/chunk-000/episode_000000.parquet")
    _ss = arr(_s, "observation.state")
    y_of_seg = {k: float(np.median(_ss[BOUND[k]:BOUND[k + 1], 19])) for k in range(1, 6)}
    print(f"  源各段 goal_y: " + "  ".join(f"seg{k}={v:+.4f}" for k, v in y_of_seg.items()))

    found = []
    for ep in (0, 1, 6, 12, 18, 23):
        dd = pd.read_parquet(DATA / f"data/chunk-000/episode_{ep:06d}.parquet")
        ds = arr(dd, "observation.state")
        # 段长必须仍是 [366, 400×4, 135]
        runs, cur, cnt = [], ds[0, 21], 0
        for v in ds[:, 21]:
            if v == cur:
                cnt += 1
            else:
                runs.append((int(cur), cnt)); cur, cnt = v, 1
        runs.append((int(cur), cnt))
        want_lens = {1: 366, 6: 135}
        len_ok = all(want_lens.get(i, 400) == L for i, L in runs) and len(runs) == 6

        # 反推排列：slot 2..5 的 dim19 中位数 → 源里哪一段
        recovered, cur = [], BOUND[1] - BOUND[0]
        for slot in range(2, 6):
            L = 400
            y = float(np.median(ds[cur:cur + L, 19])); cur += L
            k = min(y_of_seg, key=lambda kk: abs(y_of_seg[kk] - y))
            recovered.append(k if abs(y_of_seg[k] - y) < 1e-3 else -1)
        want = list(PERMS[ep % 24])
        ok = len_ok and recovered == want
        print(f"  ep{ep:03d} 反推排列 {recovered} 应为 {want}  "
              f"{'✓' if ok else '✗'}{'' if len_ok else ' (段长不对)'}")
        if not ok:
            errs.append(f"ep{ep} 反推排列 {recovered} != 设计值 {want}（段长ok={len_ok}）")
        found.append(tuple(recovered))
    if len(set(found)) == len(found):
        print(f"  ✓ 抽检的 {len(found)} 集反推排列互不相同（重排确实生效）")
    else:
        errs.append(f"抽检的集反推排列有重复: {found}")

    # 每个视频流的帧数必须 == parquet 行数（抽 1 集）
    import subprocess
    for cam in ("observation.images.cam_head", "observation.images.cam_wrist_r"):
        vp = DATA / f"videos/chunk-000/{cam}/episode_000000.mp4"
        if not vp.is_file():
            errs.append(f"缺视频 {vp}")
            continue
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries", "stream=nb_read_frames",
                            "-of", "default=nw=1:nk=1", str(vp)],
                           capture_output=True, text=True)
        got = int(r.stdout.strip())
        print(f"  {cam.split('.')[-1]} 帧数 {got} {'✓' if got == 2101 else '✗ 应为 2101'}")
        if got != 2101:
            errs.append(f"{cam} 帧数 {got} != 2101")

print(f"\n✓ 热启动源存在: {wl}")
for e in errs:
    print(f"✗ {e}")
sys.exit(1 if errs else 0)
PYEOF

echo "  （数据集怎么造的/怎么验的见 scripts/local/build_phase_shuffled_dataset.py"
echo "    与 verify_phase_shuffled_dataset.py）"
echo "GPU 现状: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)  分配器: $ALLOC_NOTE"
echo "======================================"

case "$MODE" in
  smoke)
    TOTAL=20
    EXTRA=(--num_train_steps 20 --lr_schedule.warmup_steps 5 --lr_schedule.decay_steps 20 --overwrite)
    EXP="clear5_d22idxS_smoke"; LOG="/home/dgl/lxc/.runlogs/train_d22idxS_smoke.log"
    echo "===== d22idxS 冒烟(20步) $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
  *)
    TOTAL="$MODE"; EXTRA=(--num_train_steps "$TOTAL" --lr_schedule.decay_steps "$TOTAL")
    echo "===== d22idxS 起训 $TOTAL 步 $(date '+%F %T') =====" | tee -a "$LOG"
    ;;
esac

echo "config=$CONFIG exp=$EXP batch=4 total_steps=$TOTAL mode=$MODE alloc=$ALLOC_NOTE" | tee -a "$LOG"

CKDIR="/home/dgl/lxc/openpi/checkpoints/${CONFIG}/${EXP}"
if [ -d "$CKDIR" ] && [ -z "$(ls -A "$CKDIR")" ]; then
    rmdir "$CKDIR" && echo "已清掉空的存档目录(上次崩溃残留)" | tee -a "$LOG"
fi

# 必须把 num_train_steps 和 decay_steps 一起传：只改步数不改 decay_steps 的话，
# 跑超过 config 里那个 decay_steps 之后 LR 会一直贴在地板上。
nohup "$PY" scripts/train.py "$CONFIG" \
    --exp_name "$EXP" --no-wandb_enabled \
    --batch_size 4 "${EXTRA[@]}" \
    >> "$LOG" 2>&1 &
echo $! > /tmp/d22idxS.pid
echo "PID $(cat /tmp/d22idxS.pid)"
echo "日志: $LOG"
echo "盯这行: tail -f $LOG"
