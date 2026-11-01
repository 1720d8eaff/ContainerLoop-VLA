# G1-D 双臂双夹爪 GR00T N1.7 适配

状态和动作各 16 维：左臂 7、右臂 7、左夹爪 1、右夹爪 1。它提供 GR00T 官方格式的本体配置、数据字段映射和推理数据适配；

## 文件用途

| 文件 | 用途 |
| --- | --- |
| `g1d_config.py` | 传给官方训练入口 `--modality-config-path`，注册 `NEW_EMBODIMENT` |
| `meta/modality.json` | 放入最终训练数据集的 `meta/`，定义 16 维切片、视频和语言字段 |
| `g1d_adapter.py` | 实际示教字段到 16 维数组、官方推理观测、动作块解码 |
| `robot_profile.json` | 双臂顺序、DDS索引、来源、已知与待确认的硬件参数 |
| `inspect_trajectory.py` | 检查你已有 `trajectory.json`，可导出中间 NPZ |
| `probe_gripper_state.py` | 在机载端只订阅候选夹爪状态，补充缺失的夹爪通信证据 |
| `validation_report.json` | 本次实际完成的离线检查和示教样本核查结果 |

## 本体与动作约定

| 切片 | 字段 | 示教来源 | 含义 |
| --- | --- | --- | --- |
| `[0:7]` | `left_arm` | `left_arm` | 左臂绝对关节角，rad |
| `[7:14]` | `right_arm` | `right_arm` | 右臂绝对关节角，rad |
| `[14:15]` | `left_gripper` | `left_ee` | 左夹爪原始单通道 qpos |
| `[15:16]` | `right_gripper` | `right_ee` | 右夹爪原始单通道 qpos |

每个手臂按 shoulder_pitch、shoulder_roll、shoulder_yaw、elbow、wrist_roll、wrist_pitch、wrist_yaw 排列。手册 §13.4 和本地官方 G1-D SDK 支持左臂 DDS ID 15–21、右臂 22–28；这是 SDK 顺序。示教导出器是否采用完全相同的臂内顺序，仍需与采集程序核对，不能仅靠向量长度断言正确。

四组动作全部选择 `ABSOLUTE + NON_EEF + DEFAULT`。训练标签是采集器保存的 `actions_qpos`，观测是 `states_qpos`；不要用下一帧观测代替动作，也不要把实际状态误作控制目标。
保留夹爪原始数值，不强制缩放到 `[0,1]`，不当作毫米，不假定 0 表示张开或闭合，不套用 RH56H1 六通道范围。由 GR00T 根据真实训练集计算统计并进行模型侧归一化；官方 `get_action()` 已输出还原后的动作，不要二次反归一化，也不要加上当前关节角。

## 相机和时间

默认视频键 `color_3` 对应最终数据列 `observation.images.color_3`，因为检查过的容器示教只有该视角。

`make_observation` 要求 RGB uint8 `[H,W,3]`；OpenCV BGR 需先显式转换。训练与推理应使用一致的视角、裁剪和缩放。公开样本裁掉图像上方 96 像素并缩放，不能与未处理实时原图无说明地混用。

观测历史为当前帧，动作预测长度为 16，这是本次设计选择。实际训练数据 FPS 决定 16 步覆盖多长时间，不代表 500 Hz 低层发布频率。公开样例每隔 6 帧抽样、保留末帧，时间由名义 30 FPS 换算，末尾间隔可能不均匀；不能把它直接当作完整、同步的 30 Hz 训练集。正式训练应使用完整同步示范并明确动作时间对应关系。

## 使用官方训练入口



数据中 `observation.state` 与 `action` 均为上述顺序的 16 维数组。任务列 `task_index` 和 `annotation.human.task_description` 保存真实任务在 `tasks.jsonl` 中的编号；视频列为 `observation.images.color_3`。

在官方仓库根目录运行，路径按实际位置替换：

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/g1d_dual_gripper_lerobot_v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /path/to/G1D_GR00T_adapter/g1d_config.py \
  --num-gpus 1 \
  --global-batch-size 8 \
  --max-steps 2000 \
  --save-steps 500 \
  --output-dir /data/checkpoints/g1d_dual_gripper
```

这里 batch size、2000 步和保存间隔是启动示例，需按显存与数据量调整，不是已完成训练或性能保证。一个 Python 进程只导入一份 `NEW_EMBODIMENT` 配置；不要再同时导入 SO100 示例。

## 推理数据适配

在微调权重对应的环境中使用：

```python
from gr00t.policy import Gr00tPolicy
from g1d_adapter import make_observation, decode_action_chunk

policy = Gr00tPolicy(
    model_path="/data/checkpoints/g1d_dual_gripper/checkpoint-2000",
    embodiment_tag="NEW_EMBODIMENT",
    device="cuda",
    strict=True,
)
policy.reset()
# rgb、states_qpos、instruction 由真实采集程序提供并完成时间对齐。
obs = make_observation(rgb, states_qpos, instruction)
actions, info = policy.get_action(obs)
absolute_targets = decode_action_chunk(actions)  # [预测步数, 16]
```

模型输出不直接等于经过限位、碰撞和速度检查的运动轨迹。执行层应沿用现有 G1-D 后端，复核工程 URDF 的双臂限位，并补齐夹爪实际通信绑定。`robot_profile.json` 的右臂限位已从现有工程 URDF 提取，未把左臂数值直接镜像过去；这些仍是工程参数，不是本次实机限位验收。包内解码器只返回数据，不向机器人发送命令。

## 检查现有示教

在已有 numpy 的 Python 环境中：

```bash
python inspect_trajectory.py /path/to/episode_0002/trajectory.json
python inspect_trajectory.py /path/to/episode_0002/trajectory.json --output /path/to/new_episode.npz
```

输出会保留名义时间来源和抽样间隔。NPZ 是字段核对与后续转换的中间产物，不是可以直接传给官方训练入口的 LeRobot 数据集。数据转换还需根据完整原始录像、实测时间戳与任务指令实现，不能拿三个展示子集冒充 300 条训练示范。

## 补齐夹爪通信信息

桌面 `G1_Dex1-1_SimReady_调研包` 使用普通 G1 的 Dex1-1 参考模型，结合样例图像只能形成型号候选，不能认证当前 G1-D 的控制协议。

宇树官方遥操作代码提供两类不同接法：外部 Dex1 服务使用 `rt/dex1/left/state`、`rt/dex1/right/state`；普通 G1 的内部接法使用低层数组候选槽 31/33。后者不能未经核对直接套到 G1-D。两类候选分别记录在 `robot_profile.json`，都未设为当前已确认的执行链。

可将以下脚本放到机载现有 SDK Python 环境中，先只读外部服务候选：

```bash
python probe_gripper_state.py --interface eth0 --layout external --seconds 5
```

若采集/控制源码明确使用内部接法，再只读对应候选：

```bash
python probe_gripper_state.py --interface eth0 --layout internal --seconds 5
```

脚本只有订阅器，没有发布器。收到零值也算有效数值，不以“非零”作为在线条件。收到状态证明该话题存在，不能单独证明其对应你安装的实体夹爪；需要结合现成驱动源码和实际反馈确认。没有消息可能是服务、网络或接法差异，不应自动切换到另一套发送命令的驱动。
