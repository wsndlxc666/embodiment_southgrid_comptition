# g1_omnipicker 工具收纳 —— 现在到底该跑哪个脚本

写这份是因为脚本太多、名字太像，看的人（包括我自己）会搞不清哪个才是在用的。
**下面这张表就是全部。**表里没有的，都已经被收进 `SouthGrid/_attic/`（没删，可还原，见文末）。

> **想看"模型输出的那堆数是怎么一步步变成关节力矩的"** → 看
> [README_数据流_看这里.md](README_数据流_看这里.md)。那篇讲中间过程，这篇讲跑哪个脚本。

---

## 一、四个真正在用的入口

| 我要干什么 | 跑这个 | 在哪 |
|---|---|---|
| **① 起策略服务**（模型推理） | `scripts/serve_policy.py` | `openpi/` |
| **② 闭环自测**（不看分，只看策略在仿真里收得怎么样） | `g1_omnipicker_policy_clear_eval.py` | `dataCollection/g1_omnipicker/` |
| **③ 读②的中间决策**（每次推理想去哪、夹爪何时合） | `analyze_chunks.py` | `inference/g1_omnipicker/` |
| **④ 正式评分跑**（比赛 entry，会连中央评分服务） | `run_task3_tool_storage_scored.py` | `inference/g1_omnipicker/` |
| **⑤ 复核②收到的到底是几把**（看终局坐标，不信任 `[已入箱]` 计数） | `check_final_positions.py` | `inference/g1_omnipicker/` |
| **⑥ 看中间产物**（归一化后的 state / 离散整数 / 喂给模型的句子 / 模型原始输出） | `reconstruct_intermediate.py` | `inference/g1_omnipicker/` |

**②③⑤⑥ 是一组**：②加 `--dump_chunks` 写 JSONL，③⑤⑥ 读它。③⑤⑥ 不出分、不连服务，纯离线读文件。
③ 看"每步想干嘛"，⑤ 看"最后收了几把"，⑥ 看"中间数据变成了什么样"。
⑥ 重建的每一步为什么能精确重建、哪两样恢复不出来，见 [README_数据流_看这里.md](README_数据流_看这里.md) 第四节。

### 完整跑一次（自测，不评分）

```bash
# 1) 起策略服务（另开一个终端，保持前台）
cd /home/dgl/lxc/openpi
uv run scripts/serve_policy.py pi05_g1_omnipicker_clear5_d19goal_lora_20k \
    --exp-name=clear5_d19goal_lora_100ep --checkpoint-step=9000 --port 8010

# 2) 闭环自测 + 转储（仿真侧）
cd /home/dgl/lxc/SouthGrid/src/examples/dataCollection/g1_omnipicker
python g1_omnipicker_policy_clear_eval.py \
    --port 8010 --trials 1 --sim_budget_s 130 \
    --dump_chunks /home/dgl/lxc/.runlogs/xxx_dump.jsonl --dump_echo brief

# 3) 读决策
cd /home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
python analyze_chunks.py /home/dgl/lxc/.runlogs/xxx_dump.jsonl
```

---

## 二、另外两个不能动、但不用你手动跑的

| 文件 | 为什么不能动 |
|---|---|
| `eval_g1_omnipicker_tool_lerobot.py` | `run_task3_tool_storage_scored.py` 第 34 行 `import ... as E` 依赖它。也是"整块动作依次下发"的参考写法。 |
| `g1_omnipicker_collection_scripted_tool_lerobot.py`<br>`g1_omnipicker_collect_clear5_lerobot.py` | 上面②第 75-76 行 `import ... as _T / as C5` 依赖它们（工具名表、种植/校验、箱底标定、prompt 词族都在这）。前者也是采数据的脚本。 |

参照点常量（训练/推理必须同一组）：
`inference/g1_omnipicker/tool_refs_d19goal.json`（6 点，v2）—— 被②③④都读。
`tool_refs_d19.json`（5 点，v1）留着只是为了让读 v1 的旧代码不报错，**新代码不要再用**。

---

## 二·五、state 是 19 维，别让它掉成 18

pi0.5 **不用张量吃 state，而是把它离散成文字塞进 prompt**：
`Task: 把桌面上的工具都收进工具箱, State: -1 -1 255 … 127;\nAction: `。
所以 state 少一维不是"某维被置零"，而是**模型看到一条它从没见过的串**。

训练那档（`pi05_g1_omnipicker_clear5_d19goal_lora_20k`）的 `norm_stats.json` 里
`state.mean` 长度 = **19**。②必须发 19 维，否则整轮结果都不能用来判断策略好坏。

* ②默认已经发 19 维（`--tool_refs` 默认指向 v2 json 的 `goals` 6 点），
  启动时会打一行 `[state] 19 维（第19维 goal_dist 参照点 6 个…）`，**先核对这行**。
* 传 `--tool_refs ""` 会退回 18 维 —— 只为和旧结果对照，不要用它评估策略。
* ④一直发的是 19 维（它本来就传了 refs）。所以**历史上②和④连 state 维度都不一样**。
* 自查方法：`python -c "import json;print(len(json.load(open('<ckpt>/assets/<repo>/norm_stats.json'))['norm_stats']['state']['mean']))"`

---

## 二·六、左臂冻结与布局固定（②的两个默认行为）

训练集里**左臂根本没动过**：state 的 0..6 维整集跨度只有 1.4~18 mm（右臂是 30~93 cm），
左爪两维（14,15）跨度**恰好为 0**（q01 == q99 == 0.0480 = 一直张开）。

推理时模型**原始输出**实测（跨三轮、N=135~600 条 chunk）：左臂位置跨度 15~27 mm、
四元数分量跨度 0.045~0.14，左爪跨度**恰好 0**。即它没到"乱甩"的程度，
但比训练时飘得多一点。

所以②默认**冻结左臂**：模型输出里的 `[0:7]`（位姿）+ `[14:16]`（左爪）被换成常量，
只有右臂听模型的。理由就这一条：**左臂不是这条策略学过要控制的东西**，
把一块它没学会、却仍在动的自由度钉死，总不会更差。

> **冻结改的是「指令」，不是「关节」—— 别指望它让左臂纹丝不动。**（2026-09-14 实测）
>
> | | 左臂 x 跨度 | 左臂 z 跨度 | state 标准差 (x/y/z) |
> |---|---|---|---|
> | 冻结 | 43 mm | 51 mm | 7.0 / 0.9 / 7.8 mm |
> | 未冻结 | 40 mm | 47 mm | 6.3 / 1.4 / 7.2 mm |
>
> 冻结前后**几乎没差别**，而模型**原始输出**的左臂标准差只有 3 mm。
> 把 3 mm 的指令抖动拿掉，7 mm 的**物理抖动**纹丝不动 —— 后者不由指令驱动。
> 且它与右臂活动强度的相关性只有 **+0.03**（右臂动得最少的 10% 采样点上，
> 左臂偏移反而更大 9.4 vs 5.7 mm），所以**也不是**右臂反作用力晃的。**成因未定。**
>
> 量法：转储里 `state[0:7]` 是仿真量到的左臂位姿（`build_state` 读
> `/action/end/position`，带实测值），不是指令。想去掉这个抖动只能动**物理/控制侧**，
> **动指令没用**。`_LEFT_POSE_CONST` 保证的只是"下发的指令恒等于训练均值"。
>
> **⚠ OSC `kp` 各脚本不一致（2026-09-14 查）：** 采集器（生成训练数据）默认 **220**，
> ② 也是 220，但 **④ 评分跑和 `eval_g1_omnipicker_tool_lerobot.py` 默认 150** ——
> 评分时的手臂比训练数据软 **1.47 倍**。上限 300。这个不一致之前没人注意过。
>
> 注：④ `run_task3_tool_storage_scored.py` 的 `lock_left_arm` 做的是**同一件事**
> （每步把常量写进 `l_arm.update_action_position`），所以它有完全一样的表现。

> **别再拿"减小底座反作用力"当冻结的理由**（这条 2026-09-14 被推翻）。
> 同一段代码量三个转储的**全部** 135 条记录：
>
> | 转储 | 反推底座偏航中位 | base 自身移动 (mm) |
> |---|---|---|
> | 未冻结 `clear5_d19state_0914` | −0.88° | [1.77, 11.57, 0.04] |
> | 冻结 `clear5_freeze_fixed_0914` | −1.61° | [2.42, 19.80, 0.06] |
> | 冻结 `clear5_diag_0914` | −1.21° | [2.81, 16.66, 0.11] |
>
> 冻结的两次**不比**未冻结的小，反而略大。且这个量主要由**右臂**摆动决定，两轮之间
> 右臂轨迹本就不同 → 三个样本、无固定随机种子，**两个方向都证明不了**。
> （早先报过的 "−0.33° / −3.32 mm" 只在某个子窗口上成立，全量复算不成立，已作废。）

* 常量在 [g1_omnipicker_policy_clear_eval.py](dataCollection/g1_omnipicker/g1_omnipicker_policy_clear_eval.py) 的
  `_LEFT_POSE_CONST` / `_LEFT_GRIP_CONST`，取自训练那档 `norm_stats.json` 的 `state.mean[0:7]`
  与 `q01[14]`，和仿真默认位姿差 <3 mm。
* 替换发生在 `PolicyRolloutDevice._apply()`（下发前），**不在收到 chunk 时** ——
  这样 `--dump_chunks` 里存的仍是**模型原始输出**。想知道"模型想让左臂干嘛"看转储，
  "实际下发什么"看 `_apply`，两者不混。
* 转储每行带 `"freeze_left": true/false` 自述，别把冻结前后的两次结果混着比。
* 只为对照才关：`--no_freeze_left`。

**布局默认固定**：不传 `--jitter` 时所有抖动参数给 0，`_sample_clear_layout` 直接落回
`TOOL_REF_POS_B` 标称槽位、yaw=0，每试逐位一致（只差 `round(...,4)` 的 0.03 mm）。
`--jitter` 才复现训练集那种"每集布局略不同"的分布；`--j_*` / `--big_*` 只在开了它之后生效。

启动时会打两行 `[左臂]` / `[布局]` 自述，**先核对这两行**再比结果。

---

## 二·七、"入箱 N/5" 这个数**先别全信**

2026-09-14 复核时发现 `[已入箱]` 会**误报**。已核实的例子
（`clear5_freeze_fixed_0914`，报了 3/5，**真值 1/5**）：

| 报的 | 终局位置(base 系) | 真假 |
|---|---|---|
| `螺丝刀 @step13596` | `(+4.739,+11.977,−0.634)` — 被甩出场景 12.9 m、掉到地板下 | **假** |
| `扳手 @step13638` | `(+0.564,−0.089,+0.136)` — 距标称槽位 0.4 cm，全程没动过 | **假** |
| `手电筒 @step25914` | `(+0.648,+0.094,+0.129)` — 确实在箱内 | 真 |

**同一配置（冻结 + 固定布局）换一轮跑（`clear5_diag_0914`，带理由打印）就一个误报都没有**：
报 2/5，终局 手电筒 `(+0.600,+0.080,+0.144)`、电工刀(右) `(+0.608,+0.202,+0.124)` 两把都在箱里，
另外三把分别在距槽位 0.1 / 4.1 / 0.7 cm —— 报的和真的是同一批。
**所以这个误报是偶发、不是必现**，别因为它就否定某个配置。

`螺丝刀 @13596` 那次**至今没找到成因**：翻那一轮的转储，标记发生的时刻螺丝刀正**举在半空**
（base 系 z≈+0.33，静止高度是 0.136），xy 也在判定区外 —— 几何兜底那条路走不通；
而那一轮没打理由，接触查询的结果也没进转储，**无法从现有数据定案**。
（能确定的只有：同轮 42 步后扳手被连坐标记，而扳手从头到尾没动。）

判定链路上有两个"看着像已入箱"的口子，值得警惕：

1. `_classify_tool_place_contacts` 在**无接触记录**时不做接触检查。工具静止久了会被
   物理引擎休眠、接触查询返回空，正好走进这条。
2. 兜底几何判定的 z 容差是相对 `z_surface` 的，而**工具箱底板顶面 z_surface 和工具躺在
   桌面上的高度只差几毫米** —— 于是 z 这一维几乎不设防，判定实际全压在 xy 上。
   xy 标定宽一点，桌上的工具就会被误判。

**还有一个副作用**：一旦某把工具被判成已入箱，循环就 `continue` 跳过它，不再检查 ——
所以误报不只是多加一分，还会把后面真正的失败盖住。

因此：
* 现在 `[已入箱]` 会带上**理由**（接触判定 / 几何兜底，含判定区范围和工具坐标），
  启动时也会打印**入箱判定区**和工具箱本体位置。**先看这几行** ——
  误报偶发，理由串是唯一能当场分辨的那条线索。
* 兜底几何判定已**统一到采集器自己的那份** `_T._tool_geom_in_box`（`z_tol=0.10`、
  `xy_margin=0.04`）。此前 clear_eval 里硬编码的是 `z_tol=0.15` + `xy_margin=0.02` ——
  同一个"什么算在箱里"在项目里有两套定义，而训练数据是用采集器那套标的。
  这次统一**还没在整轮里跑过**，只在四个已知案例上单点验证过（真落定的电工刀→True、
  下落中那一瞬→False、手电筒被标记时→True、上一轮误报的扳手→False）。
* 真正想知道"收了几把"，要看**终局工具坐标**（转储最后一条的 `tools_world` 减 `base_world`），
  对照 5 个标称槽位和箱子位置，而不是只看那个计数。上面两轮都是这样复核的。
  已固化成脚本（表⑤），别每次手算：
  ```bash
  cd /home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker
  python check_final_positions.py /home/dgl/lxc/.runlogs/xxx_dump.jsonl [第二个转储 ...]
  ```
  判定：`在箱里 ✓` 才算；`没动` 是**否定性**证据 —— 工具没动却出现在 `[已入箱]` 里，那必是误报。

---

## 三、回放节奏：最容易被搞错的一件事

**训练集 1 帧 = 0.05 s 仿真时间 = 10 个 env 步**（采集器 `clock="sim"` + `--fps 20`，
env_dt = 0.001×5 = 0.005 s）。一集 2101 帧 = **105 s 仿真时间**，从左到右收完 5 把。

所以正确的回放 = **每个模型动作保持 10 个 env 步，整块 20 步依次下发**
（20 × 0.05 s = 1.0 s 仿真时间，与训练同速）。②的默认值就是这个
（`--exec_mode chunk --action_repeat 10`），启动时会打印一行 `[回放]` 让你核对。

`--exec_mode first` 是旧行为（只下发第 0 步、其余 19 步丢弃），留着只为和历史结果对照，
**不要用它判断策略好坏**：单步位移只有 19 mm，而服务端每次推理都重采样 flow-matching
噪声（sd 13~30 mm），信号和噪声同量级，等于原地随机游走。

---

## 四、被收进 `_attic/` 的（**没删**）

| 文件 | 为什么收起来 |
|---|---|
| `run_task3_tool_storage_{cycle,generic,multigrasp,scripted_scored}.py` | ④的历史版本，全项目零引用 |
| `eval_g1_omnipicker_button_lerobot.py` | 按钮任务，和工具收纳无关，零引用 |
| `probe_prompt_sensitivity.py` | 一次性判据实验，结论已写进 memory |
| `g1_omnipicker_policy_{grasp,place}_eval.py` | ②的前两代（同一条血脉），已被②取代；且它们都是"只下发第 0 步"的写法 |
| `_probe_*.py` `_scan_placeable.py` `_listbodies.py` | 采数据时的临时探针，零引用 |

还原（把某个文件搬回去）：

```bash
mv /home/dgl/lxc/SouthGrid/_attic/inference_g1_omnipicker/<文件名> \
   /home/dgl/lxc/SouthGrid/src/examples/inference/g1_omnipicker/
```

确认都不要了，直接删：

```bash
rm -rf /home/dgl/lxc/SouthGrid/_attic
```

**注**：`dataCollection/g1_omnipicker/REPORT_single_grasp_2026-09-06.md` 和
`REPORT_pick_place_2026-09-06.md` 里提到的 `grasp_eval`/`place_eval` 现在在 `_attic/` 里。
这两份报告是按**旧的夹爪约定**写的，其中"夹爪反相把工具推走"的结论**已被推翻**
（正确约定：norm 1.0 = 闭合/抓、0.048 = 张开/放），读的时候注意。
