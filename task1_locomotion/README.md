# Orca HEFT G1 + Dex3-1

HEFT G1 PMG 在 OrcaLab 的专用运行包。仓库只保留 G1 + Dex3-1 播放所需的代码、ONNX 策略、动作片段、安装脚本和上游许可证。

策略动作维度固定为 29，只控制 G1 本体。Dex3-1 的 14 个 actuator 不进入策略输入输出，运行时双手保持被动张开，避免手指扰动本体控制。

本仓库同时提供 Task1（移动巡检 A/B/C 检查点）的完整评分系统接入示例，帮助你一步一步完成安装、验证、评分上报。

---

## 1. 环境要求

- 操作系统：Windows 11 / Ubuntu 22.04 / Ubuntu 24.04
- Python：3.12 及以上
- OrcaLab / OrcaStudio：26.7.1（兼容 26.6.3 及以上）
- 建议：使用 `orcalab` conda 环境

---

## 2. 安装

### 2.1 克隆仓库

```bash
git clone --branch master https://codehub.devcloud.cn-east-3.huaweicloud.com/bbb453d36204472ca668f7c8ca786e59/SouthGrid_Unitree_g1_locomotion.git
cd SouthGrid_Unitree_g1_locomotion
```

### 2.2 安装 HEFT 运行环境

默认使用 `orcalab` conda 环境（Python 3.12）。在仓库根目录执行：

```bash
conda run -n orcalab pip install -r requirements.txt
conda run -n orcalab pip install --no-deps -e .
```

安装完成后校验资产与运行时依赖：

```bash
./scripts/check_heft_install.sh --runtime
```

`check_heft_install.sh` 会自动检测 `orcalab` conda 环境；如需使用其他 Python，可设置 `ORCA_HEFT_PYTHON`：

```bash
ORCA_HEFT_PYTHON=/path/to/python3.12 ./scripts/check_heft_install.sh --runtime
```

### 2.3 安装大赛评分客户端

```bash
conda run -n orcalab pip install git+https://codehub.devcloud.cn-east-3.huaweicloud.com/bbb453d36204472ca668f7c8ca786e59/competition_scoring_client.git
```

验证安装：

```bash
conda run -n orcalab python -c "from orca_scorer_client import ScorerClient; print('OK')"
```

### 2.4 视频录制环境准备

Task1 评分会自动录制 ORCA 窗口视频并上传。评分客户端的 Python 依赖（`opencv-python`、`imageio-ffmpeg`、`mss`、`pyautogui`、`pywin32`、`python-xlib` 等）已在 `pip install` 时自动安装，只需再确认平台相关的系统依赖：

**Windows 11**

- 运行评分脚本时，ORCA 仿真窗口需保持可见，不能最小化或被其他窗口完全遮挡。
- 若使用高 DPI 显示器，Windows 已自动启用 DPI 感知，无需额外配置。

**Ubuntu 22.04 / 24.04**

- 必须安装 `x11-utils`，否则屏幕录制只能生成仅含 MP4 头的空文件（约 0.3 KB）：

  ```bash
  sudo apt-get update
  sudo apt-get install -y x11-utils
  ```

- 运行评分脚本时，ORCA 仿真窗口需处于 X11 桌面环境中；远程 SSH 无图形界面时无法录制视频。

验证录制依赖是否就绪：

```bash
conda run -n orcalab python -c "import cv2, mss, imageio_ffmpeg; print('video deps OK')"
```

### 2.5 校验安装

```bash
./scripts/check_heft_install.sh --runtime
```

所有发布资产都有 SHA-256 校验。

---

## 3. 启动 ORCA 并加载场景

1. 打开 OrcaLab / OrcaStudio。
2. 加载 Task1 场景，场景中需包含：
   - 机器人实体：`g1_pick`（G1 + Dex3-1）
   - 检查点 site：
     - `Static_Location_A_site`
     - `Static_Location_B_site`
     - `Static_Location_C_site`
3. 在 ORCA 中选择「外部仿真程序」模式，确认机器人处于可控制状态。

如果场景中找不到 G1 机器人，脚本会报错：

```text
ValueError: Cannot find robot model in scene: G1
```

---

## 4. 基础运行：键盘遥控（可选验证）

在接入评分前，可先用键盘验证 HEFT 策略能正常驱动机器人。

```bash
./play_g1_heft.sh
```

非默认服务地址：

```bash
./play_g1_heft.sh --remote 127.0.0.1:50051
```

| 按键 | 动作 |
| --- | --- |
| `W/S`、上下方向键 | 前进/后退 |
| `A/D`、左右方向键 | 左移/右移 |
| `Z/C` | 左转/右转 |
| `F1/F2/F3` | HEFT walk1/walk2/walk3 |
| 空格 | 站立 |
| `R` | 复位 |
| `Q`、`Esc` | 退出 |

SSH / 无全局键盘权限时使用：

```bash
./play_g1_heft.sh --keyboard-backend terminal
```

固定命令联调：

```bash
./play_g1_heft.sh --keyboard-backend none --lin-vel-x 0.5 --seconds 20
```

兼容入口 `./play_g1_heft_velocity.sh` 保留。

---

## 5. Task1 移动巡检评分接入

### 5.1 先验证纯导航（不上报评分）

首次验证时建议先用 `--no-score`，确认机器人能到达 A/B/C：

**Windows（PowerShell）：**

```powershell
conda run -n orcalab python run_task1_heft_navigation.py --no-score
```

**Ubuntu 22.04/24.04（Bash）：**

```bash
conda run -n orcalab python run_task1_heft_navigation.py --no-score
```

预期输出：

```text
[nav] 到达 Static_Location_A_site (dist=0.57m)
[nav] 到达 Static_Location_B_site (dist=0.55m)
[nav] 到达 Static_Location_B_site_via1 (dist=0.30m)
[nav] 到达 Static_Location_B_site_via2 (dist=0.30m)
[nav] 到达 Static_Location_B_site_via3 (dist=0.30m)
[nav] 到达 Static_Location_B_site_via4 (dist=0.29m)
[nav] 到达 Static_Location_C_site (dist=0.60m)
[heft-task1] 导航完成: {'reached': [...], 'total_steps': 7182, 'elapsed_s': 143.64}
```

> 当前默认速度较慢（`FORWARD_SPEED = 0.20 m/s`），以保证双足行走自然、不碰撞障碍物。A 点距离起点约 28 m，完整导航约需 2~3 分钟，请耐心等待。

### 5.2 正式接入评分系统

`ScorerClient` 会自动检测本地评分服务；如果未运行，会自动启动本地评分服务，自动录制 ORCA 窗口视频并上传到中央评分服务器。Task1 的机器人实体名称为 `g1_pick`，对应评分客户端里的 `g1_pick` 机器人配置，因此需要确保 `robot_id` 配置正确。

**Windows（PowerShell）：**

```powershell
conda run -n orcalab python run_task1_heft_navigation.py `
  --team-id team_005 `
  --team-token token_team_005_6116787636331575
```

**Ubuntu 22.04/24.04（Bash）：**

```bash
conda run -n orcalab python run_task1_heft_navigation.py \
  --team-id team_005 \
  --team-token token_team_005_6116787636331575
```

`run_task1_heft_navigation.py` 默认使用 `--robot-id g1_pick`，脚本在初始化 `ScorerClient` 时会把它传递给本地评分服务。

### 5.3 查看结果

脚本运行结束后，控制台会输出：

```text
[scorer] score: 20.0 / 20.0
  start_move: PASS
  pass_checkpoint_a: PASS
  pass_checkpoint_b: PASS
  arrive_panel: PASS
  smooth_motion: PASS
[scorer] 等待 20s 让视频上传完成...
```

然后登录中央评分后台，确认 attempt 和视频已上传。

---

## 6. 配置 `robot_id`

如果将来换成其他机器人形态，可以通过以下任一方式配置 `robot_id`：

### 6.1 命令行参数

脚本内已默认 `--robot-id g1_pick`：

```bash
conda run -n orcalab python run_task1_heft_navigation.py \
  --team-id team_005 \
  --team-token token_team_005_6116787636331575 \
  --robot-id g1_pick
```

### 6.2 环境变量

- Windows：
  ```powershell
  $env:ORCA_SCORING_ROBOT_ID = "g1_pick"
  ```
- Ubuntu：
  ```bash
  export ORCA_SCORING_ROBOT_ID=g1_pick
  ```

### 6.3 配置文件 `orca_scoring.yaml`

评分客户端首次运行时会自动生成该文件，通常在：

- Windows：`%APPDATA%\orca_scoring\orca_scoring.yaml`
- Ubuntu：`~/.config/orca_scoring/orca_scoring.yaml`

修改其中的：

```yaml
robot_id: "g1_pick"
```

> 如果 `robot_id` 配置错误（例如默认的 `g1_omnipicker`），评分服务会找不到正确的机器人实体，导致步骤全部 FAIL。

---

## 7. 失败轨迹测试

为验证评分系统对失败情况的判定是否正确，项目提供了 `run_task1_failure_tests.py`，可构造以下几种典型失败轨迹并上报到评分服务器：

- `no-b`：只到达 A 点，不经过 B、C
- `no-c`：经过 A、B，但不到达 C
- `standstill`：启动后原地站立不动
- `wrong-direction`：向远离 A/B/C 的方向行走

用法示例：

```bash
conda run -n orcalab python run_task1_failure_tests.py \
  --case no-b \
  --team-id team_005 \
  --team-token token_team_005_6116787636331575
```

预期这些失败用例的分数应低于 20.0/20.0，且对应步骤显示为 FAIL。例如 `no-b` 的参考输出：

```text
[scorer] score: 5.0 / 20.0
  start_move: FAIL
  pass_checkpoint_a: FAIL
  pass_checkpoint_b: FAIL
  arrive_panel: FAIL
  smooth_motion: PASS
```

---

## 8. 脚本中评分接入的关键代码

### 8.1 导入评分客户端

```python
from orca_scorer_client import ScorerClient
```

### 8.2 实例化

```python
scorer_kwargs = {
    "team_id": args.team_id,
    "team_token": args.team_token,
    "timeout": SCORER_TIMEOUT,
}
if args.server_url:
    scorer_kwargs["server_url"] = args.server_url
scorer = ScorerClient(**scorer_kwargs)
```

### 8.3 开始与结束尝试

```python
info = scorer.start_attempt(args.task_id)
print(f"[scorer] attempt started: {info.get('attempt_id')}")

# ... 机器人控制代码 ...

summary = scorer.finish()
print(f"[scorer] score: {summary.get('score')} / {summary.get('max_score')}")
for r in summary.get("step_results", []):
    status = "PASS" if r.get("success") else "FAIL"
    print(f"  {r.get('step_id')}: {status}")
```

### 8.4 等待视频上传

视频上传是异步的，脚本在 `finish()` 后等待 20 秒再退出，避免上传线程被提前终止：

```python
print(f"[scorer] 等待 {UPLOAD_WAIT_S}s 让视频上传完成...")
time.sleep(UPLOAD_WAIT_S)
```

---

## 9. 常见问题与排查

### 9.1 评分全 FAIL，但机器人明显经过了检查点

可能原因：本地评分服务使用了错误的机器人配置，未能正确探测到 `g1_pick`。

解决：确认手动启动本地评分服务时指定了 `--robot-id g1_pick`。

### 9.2 `finish` 返回 `None / None`

可能原因：本地评分服务没有正常启动，或 `ScorerClient` 连接到了错误的服务实例。

排查：

1. 确认 9000 端口有本地评分服务在运行：
   - Windows：
     ```powershell
     Get-NetTCPConnection -LocalPort 9000
     ```
   - Ubuntu：
     ```bash
     ss -ltnp | grep 9000
     ```
2. 查看本地评分服务终端是否有错误日志。
3. 确认 `--team-id` 和 `--team-token` 与服务器侧一致，且队伍已存在。

### 9.3 服务器侧看不到成绩

排查：

1. 确认本地评分服务启动命令里的 `--server-url` 指向正确的中央服务器。
2. 确认队伍账号在中央服务器上已创建。
3. 检查本地评分服务终端是否有 `401 Unauthorized` 等报错。

### 9.4 连接评分服务失败

错误示例：

```text
[scorer] score: None / None
```

排查：

1. 确认本地评分服务终端没有关闭。
2. 检查是否被防火墙拦截。
3. 检查是否有其他进程占用了 9000 端口。

### 9.5 HEFT 机器人原地踏步、不前进

HEFT 录制的 `heft_forward` 动作在 ORCA 中表现为原地踏步。本脚本通过 free joint 位姿积分驱动底盘前进，因此这是预期设计。

### 9.6 某个检查点超时未到达

查看日志中 `dist` 是否持续减小。若接近目标但步数耗尽，可增大 `run_navigation_loop` 中的 `MAX_STEPS_PER_WP`；若转向反复摆动，可增大 `ALIGN_THRESHOLD` 或降低 `YAW_CMD`。

### 9.7 找不到 `orca_scorer_client`

确认已通过 pip 安装评分客户端：

```bash
conda run -n orcalab python -c "from orca_scorer_client import ScorerClient; print('OK')"
```

---

## 10. 验证流程清单

- [ ] ORCA 已启动并加载包含 G1 与 A/B/C 检查点的场景
- [ ] `orcalab` conda 环境可用，且 `orca_scorer_client` 已安装
- [ ] `robot_id` 已配置为 `g1_pick`（脚本默认已设置）
- [ ] 先运行 `--no-score` 验证 A/B/C 均可到达
- [ ] 再运行正式评分命令
- [ ] 控制台输出 `score: 20.0 / 20.0` 且所有 step 为 PASS
- [ ] 登录中央评分后台确认 attempt 和视频已上传（可选）

---

## 11. 扩展：接入其他任务

本模式不仅适用于 Task1。若要接入 Task2（按钮操作）或 Task3（工具归纳），只需：

1. 保留 `ScorerClient` 的初始化与 `start_attempt/finish` 调用
2. 将 `args.task_id` 改为对应任务，例如 `task2_button_press`
3. 根据该任务的机器人配置，通过 `--robot-id`、环境变量或配置文件指定正确的机器人形态

核心原则：**控制代码保持独立，评分只通过 `start_attempt` 和 `finish` 两个 HTTP 调用接入。**

---

## 12. HEFT 引用

本运行包内的 G1 PMG 策略和 `walk1/walk2/walk3` 动作源自 [Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) 的 `sim2real` 分支，固定版本为 [`0d5ba31e33397f3543d350d98b637e26d92f470a`](https://github.com/Axellwppr/motion_tracking/commit/0d5ba31e33397f3543d350d98b637e26d92f470a)，按 MIT License 使用；Copyright (c) 2026 Axell。

完整部署与排障另见 [`docs/HEFT_DEX3.md`](docs/HEFT_DEX3.md)。
