# 早上看这里（2026-09-21）

昨晚做了三件事：**定位根因 → 修 harness → 起一次过夜训练**。
下面按"先做什么、看哪个数、什么算好"排。

> **训练已跑完（`04:29:47` 干净退出）。** PID `1356680`，`22:42:41` 起，全程 `1.73 s/it`，
> 总耗时 `5:45:33`。**GPU 已释放（638 MiB 空闲）**，仿真/服务器现在都能起，
> 第 1/2/3/5 件全部解锁。
>
> - 落盘档：**`5000` / `10000` / `11999`** 三个（`keep_period=5000`，中间每 1000 步档
>   落过盘但被 orbax 回收了 —— 别去找 step 1000）。三个都完整：`params` + 
>   `assets/g1_omnipicker_clear5_d22_100ep/norm_stats.json`，各 8.7~8.8G。
> - **最终档号是 `11999` 不是 `12000`**（最后一步存档落在 `num_train_steps-1`）。
>   `serve_idxS.sh` / `morning_probe_idxS.sh` 不给步数都自动取最新，不用手打。
> - 日志 `/home/dgl/lxc/.runlogs/train_d22idxS.log`；逐维诊断
>   `openpi/data/g1_omnipicker_clear5_d22idxS_100ep/train_diag_idxS.jsonl`。
> - 全量日志扫描**无 error/NaN/traceback**：只有两条无害的（无 TPU 后端、orbax 的
>   `EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID` 取不到值）。

---

## 一、根因（一句话）

**上一轮不是模型推理停了，是 harness 把 `cycle_idx` 在错误的时刻推进，喂给模型一个
训练里不可能出现的 state。**

两条实测证据（不是推测）：

1. **模型一直在推理。** `rollout_idxR_notl.log` 里 `step` 每 3.2s 恰好 +120（= 6 次推理），
   292s 内从 10021 单调涨到 11341，没有断口；`cam[...] miss=0`；动作块每块都在变
   （|Δact|max 1.4e-2~5.1e-2），不是零输出也不是冻结输出。
   卡住的形式是 **6 段 ≥5s 的停顿、37% 的块位移 <1mm、每块末端目标离当前 EE 只 24~33mm**
   —— 微指令互相抵消，`grip_m=+2.00`（爪闭着）。
2. **idx 在错误时刻被推。** 模型从 t=9.2s 起右爪一直闭着、**从没松开过**，
   但 official 口径在 **t=28.1s** 把 idx 从 1 推到 2。之后 idx 从 t=85.1s 冻到 t=292.0s，
   **整整 207s 卡在 3，一次都没超过 3**。
   （心跳那行 `入箱=0/5` 当时看着矛盾 —— 不矛盾：`PLACED_DIST=0.20` <
   `OFFICIAL_SCORE_DIST=0.25`，工具落在 0.20~0.25 这条带里时官方判据算得分、心跳不算。）

**idx 错的连带伤害**：`cycle_dist`(dim20) = ‖ee − cycle_refs[idx−1]‖，`goal_y`(dim19)
也一起错，这两维在 loss 里权重 3.0。训练里它们**永远**与模型当下在够的那件工具自洽。

---

## 二、改了四处

| # | 改哪 | 改成什么 | 为什么（可实测） |
|---|---|---|---|
| 1 | `run_task3_tool_storage_scored.py` `--progress_gate` **默认** | `official` → **`grip`** | 训练数据里 idx 的跳变帧恰好是右爪『闭→开』边沿的**下一帧**：8/8 集、5/5 个边界，差值恒为 +1。`lerobot_data_storage.py:1186` 的 grip 分支与采集端 `build_d22_cycle_dataset.py` 逐字对齐 ⇒ 只有 grip 让相位演化与训练同分布 |
| 2 | 同上，cycle 停止条件 | `cyc_idx>=6` **且** `n_in>=1` 才收摊 | grip 下 idx 由模型自己开爪驱动，**空开 5 次爪就能顶到 6**，否则会以"模型学到的停止"收尾而箱里一件没有 |
| 3 | `openpi/src/openpi/training/config.py` 新增档 | `pi05_g1_omnipicker_clear5_d22idxS_lora_12k` | 见下面第五节，**这是实验，不是主修法** |
| 4 | 同档的 `AssetsConfig` | 加 `asset_id="g1_omnipicker_clear5_d22_100ep"` | 不给就默认取 `repo_id`，新 repo_id 没建过 stats ⇒ `norm_stats=None` ⇒ 起训直接 `ValueError: Normalization stats not found`。**冒烟闸抓到的**，见 2.1 |

### 2.1 冒烟闸抓到的两个真 bug（都是我的闸写错了，不是数据错）

起训前那道 20 步冒烟，实际拦下了两个会白白浪费一整夜的问题：

1. **`norm_stats` 根本没加载** —— 换 repo_id 后 openpi 按 `asset_id`（默认 = repo_id）找
   `<assets_dir>/<asset_id>/norm_stats.json`，找不到就静默 `None`，日志里只有淡淡一行
   `Norm stats not found ..., skipping.`，几秒后才抛 `ValueError`。
   **能不能复用源那份？我不拍脑袋，算了一遍**：用 `openpi.shared.normalize.RunningStats`
   把两个数据集各扫一遍逐维比 `mean/std/q01/q99`，**全维最大差 3.2e-13**（纯浮点累加噪声），
   因为每集各段长度不变、重排是双射 ⇒ dim19/20/21 的边缘直方图一格没动。
   所以复用是**精确正确**的，还顺带保住了"只换数据源"的单变量性。
2. **我那道闸本身是假闸**：第一版读的是 `cfg.data.base_config.norm_stats` —— 那**永远**是
   `None`，因为 `norm_stats` 是 `create_base_config()` 用 `dataclasses.replace` 挂到
   **新对象**上的，而 `base_config` 是**传进去的输入**。必须查 `create()` 的产物。
   已改成 `cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats`。

> **同一个根源本轮出现三次**：判据/被查对象与真正生效的那条路径不是同一个东西。
> 另两次是 ——  `verify` 里用**原始段号**索引 `BOUND`（seg1..seg4 等长，恒等排列会把
> 下标错误掩盖掉）；以及拿 **dim21 的"段长签名"**当"重排生效"的判据（任何排列的段长序列
> 都长一样，对排列完全不变）。后者现在换成了**反推排列**：从 slot 的 dim19 中位数反查它
> 放的是源里哪一段，得到的排列必须恰好等于 `PERMS[ep%24]`。实测 6 集全中。

> 顺带修掉一个会误导人的东西：`[model]` 那条"机器人模型不是作者那份"的横幅。
> 旧代码读的是**当前位姿**的世界朝向（`_mjData.xmat`），手臂一抬就报"模型被换了"
> —— 假报警，而且跟"相机能不能用"没关系（我逐张看过 `task3_frames_idxR/` 里今天的
> 头/腕相机画面，**都是正的**）。现在改成读**静息**朝向（`_rest_world_rot`，组合
> `body_quat`），只在模型文件真的不同时才报，且说明它**不能**推出"视觉没用"。

---

## 三、早上按这个顺序做

### 第 1 件（最重要，**不用等任何训练**）：用现有 idxR/11999 跑修好的 harness

```bash
cd /home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
./morning_run_idxR_grip.sh
```

脚本会自己检查 8010/50051、缺服务器就起（`serve_idxR.sh 11999`）。
**唯一要你手动做的是 OrcaLab**：加载场景 → Publishing scene → 再进运行模式
（顺序反了会报 `KeyError 'g1_omnipicker_*_pctrl'`；判定看 `~/.orcagym/tmp/` 最新的 xml
是不是 291863 字节那种）。

**为什么先做这件**：本轮真正的主修法是 harness，不需要等训练。手里的 idxR/11999 就能验。

**看这三个数**（脚本最后会自动打印）：
- **idx 有没有超过 3** —— 上一轮冻在 3 整整 207s。`grep '\[cycle\]' 日志`。
- **停顿还在不在 idx=3 上重复** —— 脚本会跑 `analyze_stall.py`。
- **入箱几件** —— 脚本会跑 `check_final_positions.py`（按终局坐标复核，别信 `[已入箱]` 那个计数）。

### 第 2 件：换 `--start_align pose` 再跑一次（同一条命令加参数即可）

上一轮开局自检写着：`对齐完成（1.8s）：位置残差 4.92 mm **姿态残差 110.38°**`。
`--start_align pos` 只较位置，**开局就把右腕以 110° 的姿态误差送进模型**。
`pose` 档实测是 50.6mm/17.88°（位置差一点、姿态好得多）。这是**不需要训练就能试**的
另一条分布外来源，值得单列一条对照。

### 第 3 件（**不需要仿真**）：量新档有没有开始听相位通道

```bash
./morning_probe_idxS.sh
```
探针只用数据集 + 策略服务器，**不用进 OrcaLab**。它会起 idxR/11999 和 idxS 各一次服务器，
跑六条件、和对照臂比。主指标是**有符号投影 proj（mm）**，零假设严格为 0。

> ⚠ **探针的参照集默认是原始 d22**（`probe_index_channel2.py` 里的 `DS`，那张
> `CHANGE = [(366,2),(766,3),(1166,4),(1566,5)]` 的表）。而 idxS 模型**恰恰是被训练成
> 不信这张表的**（同一个 idx 值在别集里对应别的工具）。用原布局的帧去探它，问的其实是
> "它在原布局下会不会照表走"——那正是被训练掉的行为，所以读数会**偏低**，
> 对照臂 idxR/11999 反而占便宜。
> 已经加了 `PROBE_DS` 环境变量可以换参照集，但**换了之后 `CHANGE` 那张表也不再成立**
> （每集排列不同），要另行设计，别直接套用。**这一轮探针只能当参考读数，不是判决。**

### 第 4 件：看训练日志（已跑完，现在是"复盘"不是"等"）
```bash
tail -40 /home/dgl/lxc/.runlogs/train_d22idxS.log
# 逐维诊断（1202 条）：
/home/dgl/lxc/openpi/.venv/bin/python -c "
import json
for l in open('/home/dgl/lxc/openpi/data/g1_omnipicker_clear5_d22idxS_100ep/train_diag_idxS.jsonl'):
    r=json.loads(l)
    if r['step']%1000==0: print(r['step'], r['loss'], r.get('dim21'))
"
```
读数已经在 **5.1** 里整理好了。

### 第 5 件（可选，需要仿真）：新档自己的闭环
训练已经退出，**不用再停任何东西**，直接起服务器：
```bash
bash /home/dgl/lxc/openpi/scripts/local/serve_idxS.sh      # 自动取最新档 11999
cd /home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
./morning_run_idxR_grip.sh --no-serve                      # 同一套流程，跑的就是 8010 上那个档
```
`morning_run_idxR_grip.sh --no-serve` 只借用"服务器已在跑"这一点，跑的是 8010 上的策略，
所以它对新档一样适用（TAG 里还写着 idxR11999，看日志时别被名字骗了）。
**注意这条会和第 1 件抢 8010** —— 跑之前先确认上一个服务器已停。

---

## 四、什么算好、什么算坏

| 观察 | 读法 |
|---|---|
| idx 走过 4、5、6 | **主修法有效** —— 相位跟上模型自己的动作了 |
| idx 仍卡住，但卡的位置变了 | 主修法只解决了一部分，去看停顿分析是哪一步 |
| idx 连 1 都推不动 | 模型**从不松爪** ⇒ grip 口径推不动它。这是 grip 的已知代价，换回 `--progress_gate official` 对照 |
| 入箱 0 件但 idx 走完了 | 模型在"空开爪" —— 看 `analyze_chunks.py` 的"最近规划距"，判断是"瞄了但到不了"还是"根本没瞄" |
| 探针 `all ref`/`idx ref` 的 proj 比对照臂大 | 新档确实更听相位通道（**闭环分数分不出这个，只能靠探针**） |

---

## 五、过夜训练是什么（**实验，别当主修法**）

档 `pi05_g1_omnipicker_clear5_d22idxS_lora_12k` / exp `clear5_d22idxS_lora_100ep`，
热启动 `idxR/11999`（**recipe 逐字不变，只换数据源** ⇒ 单变量可归因），12000 步 ≈ 5.3h。

数据换了什么：原数据 **100/100 集的 `cycle_idx` 跳变帧完全相同**（366/766/1166/1566/1966），
每集都是 2101 帧 ⇒ **idx 在训练里是帧号的纯函数**，携带的信息不超过"图像进度"本身。
新数据集 `g1_omnipicker_clear5_d22idxS_100ep` 把每集的 seg1..seg4 按 24 种排列轮换重排，
**只改 dim21 一维**，其余 21 维逐帧逐字节等于源。

**实测重排效果**（建完后逐集复核，`(idx → 有哪些工具)`）：

| idx 位置 | 对应工具分布（24 集里） |
|---|---|
| 1 | 扳手 ×24 —— **恒定，见下面的局限** |
| 2 / 3 / 4 / 5 | 螺丝刀 / 电工刀L / 手电筒 / 电工刀R **各 6 次，完全均匀** |
| 6 | 收尾段（不抓东西） |

**⚠ 它的诚实局限，别指望太高**：
- **`idx=1` 仍然恒等于扳手**。因为 `seg0` 从 home 位姿出发、必须排第一（拼到别处
  会造 630~680mm 跳变，远超原数据单帧最大位移 111mm）。这是结构性代价，不是疏忽：
  **第 1 轮模型仍能靠 idx 直接知道去拿扳手**，只有 2~5 轮被真正打散。
- 重排**没有**打破"idx 是帧号的纯函数"（新布局里 idx 仍 = 段位 = 帧号的函数）。
  真正拆掉的只是"**位置 p ⇒ 工具 p**"这条捷径。
- 而按同一套梯度论证（见 memory `openpi-index-channel-is-weak` §3/§5）：新数据里
  **图像单独仍然充分**（看一眼就知道该拿哪件），所以指向相位通道的梯度**依然是零**，
  拆掉那条捷径很可能不影响它走图像这条线。**更糟的情形**是训练把 idx 的权重进一步压向 0。
- **所以它更像"把处方跑一遍、留个读数"，不是主修法。** 判据是第 3 件的探针，不是入箱数。

### 5.1 跑完了，读数就是这样（**别过度解读**）

`train_diag_idxS.jsonl` 逐维 loss，和对照臂 `idxR`（同 recipe、同源数据、未重排）并排：

| step | idxR 总 loss | idxS 总 loss | idxR **dim21** | idxS **dim21** |
|---|---|---|---|---|
| 0（热启动瞬间） | 0.0713 | **0.2726** | 0.0065 | **0.0975** |
| 10 | 0.0098 | 0.1209 | 0.0058 | **0.1974** |
| 100 | 0.0213 | 0.0966 | 0.0135 | **0.5404** ← 峰值 |
| 1000 | 0.0152 | 0.0187 | 0.0211 | 0.0248 |
| 5000 | 0.0297 | 0.0368 | 0.0070 | 0.0128 |
| 11900 | 0.0111 | 0.0140 | 0.0020 | **0.0045** |

怎么读：

- **热启动确实被"打了一拳"**：dim21 的 loss 从 0.0065 冲到 0.54，正是因为模型从
  `idxR/11999` 带过来的"位置 p ⇒ 工具 p"那条先验在新数据里**到处都错**。这证明重排
  在数据层面是**真的生效了**（不是我以为它生效了）—— 这是个独立的正面证据。
- **但它很快又学会了**：1000 步就回到和 idxR 同一量级，11900 步时 `0.0045` vs `0.0020`。
- **⚠ 这一条不能当成"它开始用相位通道了"。** dim21 在新数据里**仍然是图像的纯函数**
  （看一眼就知道该拿哪件），所以 loss 低只说明"模型自洽"，不说明"模型靠这一维做决定"。
  按 memory `openpi-index-channel-is-weak` §3 那套梯度论证，**指向这一维的梯度依然可以是零**。
- **总 loss 两边基本同水平**（0.0140 vs 0.0111，差值在 idxS 带着热启动冲击的背景下不显著）
  ⇒ **既没有证据说重排有害，也没有证据说它有用。** 目前它是一个中性实验。
- **真正的判决只能来自第 3 件的探针**，而探针本身对 idxS 有已知偏置（见第 3 件那个 ⚠）。
  所以更诚实的说法是：**这一夜买到的是一个确诊过的 harness bug 修复 + 一份中性读数**，
  主修法仍然是 harness，不是这个数据集。

---

## 六、我新建/改动的文件

**改动的源码**
- `run_task3_tool_storage_scored.py` —— `--progress_gate` 默认、停止护栏、`_mark_pre_scoring` 跳过、
  `[model]` 自检改为静息朝向（`_rest_world_rot`）、`--jsonl` 未给时从 `--dump_chunks` 推导
- `openpi/src/openpi/training/config.py` —— 新增 `..._d22idxS_lora_12k` 档

**新脚本**（都在 `openpi/scripts/local/`，只读或一次性）
- `measure_splice_gap.py` —— 量相位段拼接处的位姿缝隙（结论：0~105mm，原数据单帧最大位移本就 111mm ⇒ 不平滑）
- `build_phase_shuffled_dataset.py` —— 造 idxS 数据集
- `verify_phase_shuffled_dataset.py` —— 逐帧对账（除 dim21 外 21 维逐字节相等、视频帧数、拼缝 PSNR）
- `strip_parquet_meta.py` —— 抹掉分片里多余的 pandas metadata（源是 `metadata=None`）
- `train_d22idxS.sh` —— 起训脚本（带数据集硬闸 + norm_stats 闸）
- `serve_idxS.sh` —— 起 idxS 策略服务器（不给步数就取最新档；训练没停会拒绝启动）

**在这个目录**
- `morning_run_idxR_grip.sh` —— 早上第 1 件（闭环，需要你开仿真）
- `morning_probe_idxS.sh` —— 早上第 3 件（探针，**不需要仿真**）
- 本文件

---

## 七、还有一条一直没解决的老问题（没动，只是标出来）

右爪**抓不牢**：上一轮模型在离扳手 **70mm** 处就闭爪了，训练里 EE 是走到 goals[k] 的
**4~19mm** 内。所以"抓空"可能是个独立的、比相位更靠前的瓶颈 —— 相位修好了它也不会自动消失。
`analyze_chunks.py` 的"最近规划距"能把它和"够不着"分开。
