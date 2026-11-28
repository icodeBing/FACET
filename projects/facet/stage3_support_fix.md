# Stage 3：释放后的支撑姿态修正

针对 `1127_1600.csv` 中“左膝贴近 −5° 下限、右脚偏到外侧、腰部补偿”的姿态。
新奖励仅配置在 `facet/G1LocoFacetForceBackwardStand`，可从当前 Stage 3 PT 继续训练。

## 默认奖励

| 项目 | 权重 | 启用条件与目标 |
| --- | --- | --- |
| `facet_knee_extension_margin` | 1.5 | 实测脚接触地面时，惩罚对应膝角低于 0.05 rad；前进、后退、释放共用。只约束过度伸展，不跟踪固定屈膝姿态。 |
| `facet_stand_support_center` | 1.5 | 零速度且无外力持续 0.8 s 后，双脚均已接触超过 0.08 s 时逐渐启用；允许 CoM 相对双脚中点 ±4 cm 的横向偏移，0.20 s 接触后达到完整权重。 |
| `facet_stand_leg_balance` | 1.0 | 零速度且无外力期间，对最近 0.8 s 的左右膝、髋 pitch 有符号差值取平均；膝差允许 0.08 rad，髋差允许 0.10 rad。交替迈步的差值可相互抵消，持续单侧姿态仍受惩罚。 |

速度、释放位置、身体姿态和原有足部奖励继续生效。动作缩放、PD 参数、观测及 PT 接口沿用原配置。
不启用固定腿部默认姿态、强制双脚接触或禁止脚运动的奖励。

## 训练

在服务器项目根目录执行。以下路径是上次日志记录的最终 PT；如果文件位置改变，更新 `checkpoint_path`。
先增加 100M frames，约 763 轮，检查趋势后再决定是否延长。

```bash
mkdir -p training_logs

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
env -u DISPLAY -u WAYLAND_DISPLAY \
python scripts/train_ppo.py \
  --task=facet/G1LocoFacetForceBackwardStand \
  --algo=ppo \
  task.num_envs=4096 \
  total_frames=100000000 \
  checkpoint_path=/home/sbing/git_prj/FACET/scripts/wandb/run-20261126_212952-rkjnxsiq/files/checkpoint_final.pt \
  headless=true \
  wandb.mode=disabled \
  2>&1 | tee training_logs/stage3_support_fix.log
```

## 固定条件回放

更新 `FACET_PT` 为本轮生成的新权重绝对路径。一次 seed=42，每个方向录制 2000 步。
这里单独覆盖力的方向和时序，是为了前进/后退各受力一次、释放足够长，便于比较。
相机沿用 `cfg/play.yaml` 的默认配置。

```bash
FACET_PT=/绝对路径/本轮新生成的checkpoint.pt
mkdir -p outputs_play

for FACET_DIRECTION in 1 -1; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  env -u DISPLAY -u WAYLAND_DISPLAY \
  python scripts/play.py \
    --task=facet/G1LocoFacetForceBackwardStand \
    --algo=ppo \
    checkpoint_path="$FACET_PT" \
    task.num_envs=1 seed=42 headless=true \
    record_video=true record_steps=2000 record_gait_csv=true \
    show_com_trajectories=true \
    gait_csv_path="outputs_play/stage3_support_fix_${FACET_DIRECTION}.csv" \
    task.command.force_drive_direction="$FACET_DIRECTION" \
    task.command.force_drive_once=true \
    'task.command.force_drive_duration_range=[5.0,5.0]' \
    'task.command.force_drive_start_delay_range=[1.0,1.0]'
done
```

## 检查重点

- `support/knee_margin_violation_fraction`：接触地面的膝盖落入软下界的比例。
- `release/mean_knee_difference_rad`：窗口平均左右膝差，越小越好。
- `release/support_center_error_m` 与 `release/support_center_active_fraction`：在启用条件下的横向偏移和启用比例；启用比例接近零不能解释为居中已改善。
- 同时回测受力速度误差、释放窗口速度、释放点误差，避免只改善外观而损失原有能力。

CSV 新增膝目标角/跟踪误差/实际施加力矩/距模型下限余量、左右脚接触力及竖直载荷比例、CoM 支撑中心偏移。
原有手臂和腰髋跟踪列增加实际施加力矩。接触力与载荷比例是当前传感器采样值，应按阶段或时间窗口分析，不能用单帧推断长期承重。

这是奖励目标的修正，需要续训才会改变策略；仅用新代码重新回放旧 PT，只会增加诊断数据。
CPU 数值与配置测试不能替代服务器上的 Isaac 训练和回放验证。
