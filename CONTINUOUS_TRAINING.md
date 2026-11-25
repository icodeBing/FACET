# 持续训练与按需回放

训练使用 `scripts/train_ppo.py`；`total_frames=-1` 或 `null` 表示持续训练。
不写该参数仍使用配置中的有限默认值。`schedule_frames=150000000` 是默认
课程/调度进度的长度，到达后进度保持 1，不会停止训练。

以下服务器命令均在服务器的项目根目录运行，例如 `/home/sbing/git_prj/FACET`，
先激活原先可训练的 Python 环境。本地命令在本地 `active-adaptation` 根目录运行。

## 1. 服务器：持续训练

建议先运行 `tmux new -s facet_train`，避免 SSH 断开使训练退出。
替换 checkpoint 路径后执行：

```bash
mkdir -p training_logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
env -u DISPLAY -u WAYLAND_DISPLAY python -u scripts/train_ppo.py \
  --task=facet/G1LocoFacetForceBackwardStand \
  --algo=ppo \
  task.num_envs=4096 \
  total_frames=-1 \
  checkpoint_path=/实际路径/checkpoint_final.pt \
  headless=true \
  wandb.mode=disabled \
  2>&1 | tee -i training_logs/stage3_continuous.log
```

`Ctrl+B` 再按 `D` 分离 tmux；`tmux attach -t facet_train` 返回。
在训练窗口按一次 `Ctrl+C`，等待本次更新结束，打印 `Final checkpoint` 后再退出。
`tee -i` 避免 tee 先退出导致训练输出管道断裂。SIGTERM 也使用该保存流程；
强制杀进程无法保存新权重，只能使用之前已保存的文件。

默认每 4 次更新保存/覆盖临时权重，每 100 次保留编号权重。
`checkpoint_latest.pt` 始终指向最近一次完整保存；`cfg.yaml` 在同一目录。
长期训练会积累编号权重和文本日志，应按需要归档。加载 PT 继续训练
仍沿用项目原来的加载语义，不能视为完整恢复优化器和训练进度。

## 2. 服务器：启动权重下载服务

另开 SSH 窗口（需长期保留时也可使用单独的 tmux 会话）：

```bash
python scripts/checkpoint_server.py --root scripts/wandb --bind 127.0.0.1 --port 8765
```

页面列出所有日期的运行；下载地址是 `/download/run-日期时间-ID`，
配置地址是 `/config/run-日期时间-ID`。运行名从页面或训练输出
`Checkpoint directory` 获取，不能把这里的示例名当真实运行名。

## 3. 本地：建立 SSH 转发

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 用户名@服务器地址
```

该窗口保持打开。本地浏览器打开 http://127.0.0.1:8765/ 。
若 SSH 使用自定义端口，加 `-p SSH端口`。这是 SSH 转发下载权重，
不是远程传输仿真画面；本地回放仍需要可运行 Isaac Sim 的 GPU 环境。

## 4. 本地：随时获取一份最新快照并回放 2000 步

把页面显示的实际运行名填入以下变量，同一训练只需填写一次：

```bash
FACET_RUN_ID='run-实际日期时间-实际ID'
mkdir -p checkpoints
```

之后每次想看最新效果，重新执行下载命令；下载成功后再替换本地文件：

```bash
curl --fail --location \
  "http://127.0.0.1:8765/download/${FACET_RUN_ID}" \
  --output checkpoints/stage3_latest.pt.tmp && \
mv checkpoints/stage3_latest.pt.tmp checkpoints/stage3_latest.pt
```

前进受力 5 秒后释放，录制视频和 CSV：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
env -u DISPLAY -u WAYLAND_DISPLAY python scripts/play.py \
  --task=facet/G1LocoFacetForceBackwardStand \
  --algo=ppo \
  checkpoint_path="$(pwd)/checkpoints/stage3_latest.pt" \
  task.num_envs=1 \
  seed=42 \
  headless=true \
  record_video=true \
  record_gait_csv=true \
  record_steps=2000 \
  task.command.force_drive_direction=1.0 \
  task.command.force_drive_once=true \
  task.command.force_drive_start_delay_range='[1.0,1.0]' \
  task.command.force_drive_duration_range='[5.0,5.0]'
```

回测后退时把 `task.command.force_drive_direction=1.0` 改成 `-1.0`。
其余回放条件相同，便于比较。相机沿用现有默认值，无需重复输入。
视频在 `scripts/videos`，CSV 在 `outputs_play`；回放启动时的控制台会显示路径。

每次回放始终使用启动时加载的固定快照。现有 URL 后台刷新只下载文件，
没有把新权重热加载到策略；因此这里显式下载后启动新回放。
该流程不改变奖励、外力动力学或任务目标，持续训练也不会自动修复奖励设计问题。

## 验证

CPU 检查：`python scripts/check_training_runtime.py`。
包含训练预算、无限模式调度、当前/旧目录兼容、跨日期发现、HTTP GET/HEAD、
配置下载及原子替换后下载验证。Isaac Sim 训练/回放需在完整仿真环境验证。
