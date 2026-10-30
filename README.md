# ContainerLoop-VLA

面向宇树 G1-D 容器操作的物体中心几何控制、数据对齐与评测工具。代码将目标位姿表示在物体坐标系中，组合动作投影、阶段门控、反馈纠偏、IK 和关节 PD 计算，并提供相机外参拟合、时间戳同步及 CEM 动作优化接口。

独立项目：[FailureVLA](https://github.com/1720d8eaff/FailureVLA) 研究失败恢复和异步执行；[unitree-g1d-grasp](https://github.com/1720d8eaff/unitree-g1d-grasp) 提供 AnyGrasp 抓取与硬件工程。

## 运行入口

默认 `requirements.txt` 安装 CPU 几何、相机与 MuJoCo 测试依赖。原有混合模型环境清单保留在 `requirements-model-reference.txt` 供核对，不作为默认安装入口。

接触仿真：`python -m containerloop.contact_benchmark`，使用 `configs/contact_conditions.json` 的 48 个明确条件，运行五组控制开关，共 240 次。`artifacts/contact/` 保存本次新运行的逐步 CSV、结果 JSONL 和汇总。场景是带重力、摩擦和双指接触的笛卡尔夹爪搬运圆柱代理，不是 G1-D 或 GR00T 实验；结果不对应简历历史指标。

权重推理：在安装 NVIDIA Isaac-GR00T 的独立环境中运行 `python -m containerloop.checkpoint --checkpoint PATH --embodiment TAG --input observations.npz --instruction "pick container" --output actions.npz`。输入使用 `video.<camera>` 的 uint8 `[1,T,H,W,3]` 以及 `state.<stream>` 的 `[1,T,D]` 数组，模态名称和历史长度必须匹配 checkpoint。官方 `Gr00tPolicy.get_action` 已返回物理量，不能再次反归一化。接口依据 [NVIDIA 官方文档](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)；本次未加载实际 GR00T 权重。

`decode_chunk` 对显式 7D 世界坐标系动作提供均值/标准差还原、SE(3) 步长限制及时间戳。测试使用可手算输入，不冒充缺失的模型输出记录。`GROOT_CKPT`、`GROOT_INPUT` 和 `GROOT_EMBODIMENT` 三个环境变量齐备时可运行实际 checkpoint 测试。

`camera.py` 读取 RGB 与深度文件、按提供的 Brown-Conrady 内参做反投影，并报告时间间隙；时钟比例和偏置由标定结果提供。`configs/g1d_binding.json` 从现有 unitree 工程提取 URDF 哈希、关节名、DDS 索引、限位、话题和夹爪型号；500 Hz DDS 执行由该工程后端提供。本仓库的 `joint_pd` 是计算组件。

## 目录

```text
containerloop/
  geometry.py      SE(3) 位姿、误差与动作几何投影
  control.py       阶段门控、阻尼最小二乘 IK 和关节 PD
  data.py          相机外参拟合和时间戳最近邻对齐
  episodes.py      双视角观测对齐及 JSONL 输出
  optim.py         可接入仿真目标函数的 CEM 动作序列优化
  evaluation.py    成功率、位置漂移和匹配初始条件统计
  demo.py          合成点质量模型示例
tests/             离线单元测试
```

## 安装与离线运行

```bash
git clone https://github.com/1720d8eaff/ContainerLoop-VLA.git
cd ContainerLoop-VLA
python -m pip install -r requirements.txt
python -m containerloop.demo
python -m unittest discover -s tests -v
python -m pip install pytest
python -m pytest -q tests
```

示例使用合成状态，不连接机器人。所有位姿矩阵采用 `T_A_B` 约定：将 B 坐标系中的点转换到 A 坐标系；平移单位为米、旋转误差单位为弧度。

`tests/fixtures/synthetic_states.csv` 是明确标记为 `synthetic_sdv` 的少量合成测试样本；pytest 使用它调用几何投影与 IK 接口.

## 物体中心闭环

```python
import numpy as np
from containerloop.geometry import relative, world_target, pose_error, project_delta
from containerloop.control import StageGate, damped_ik_step

T_world_object = np.eye(4)      # 从感知获取
T_world_tool = np.eye(4)        # 从机器人状态和正运动学获取
T_object_goal = np.eye(4)        # 当前阶段的物体相对目标
T_object_tool = relative(T_world_object, T_world_tool)
target = world_target(T_world_object, T_object_goal)
error = pose_error(T_world_tool, target)  # 世界系 [平移, 旋转向量]
delta = project_delta(error, translation_limit=0.02, rotation_limit=0.1)
gate = StageGate(position_tolerance=0.02, angle_tolerance=0.15)
stage = gate.update(np.linalg.norm(error[:3]), np.linalg.norm(error[3:]))
# 使用与 delta 同坐标系的 6×N 雅可比和现场关节限位：
# q_next, accepted = damped_ik_step(J, delta, q, q_min, q_max,
#                                   collision_free=collision_check)
```

阶段按接近、抓取、转运、放置、完成推进；抓取和释放阶段需要外部反馈确认。`damped_ik_step` 接受碰撞检查回调，`joint_pd` 输出限幅后的关节力矩计算值。控制接口的调用方负责接入机器人状态、碰撞模型、控制频率和执行链。

## 多视角数据对齐

`rigid_calibration` 根据配对的三维标定点计算 `T_target_source` 和拟合残差；两个相机分别调用即可得到各自的外参。标定点需要至少三个非共线点。

```python
from containerloop.data import rigid_calibration
from containerloop.episodes import align_episode, write_jsonl

# T_world_camera, rms_m = rigid_calibration(camera_points, world_points)
states = [{'timestamp': 1.00, 'state': [0.1, 0.2], 'action': [0.2, 0.3], 'stage': 'approach'}]
cameras = {
    'front': [{'timestamp': 1.01, 'path': 'front/0001.png'}],
    'wrist': [{'timestamp': 0.99, 'path': 'wrist/0001.png'}],
}
rows = align_episode(states, cameras, max_offset=0.05)
write_jsonl(rows, 'runs/example/episode.jsonl')
```

各时间戳需提前统一到同一时钟，以秒为单位且按时间排序。超出容差的相机帧记为 `null`。每行可保留状态、动作、阶段和各视角文件路径；后续可按训练数据使用的字段、坐标系和采样频率转换。

## 动作优化与评测

`cem` 接收动作序列并调用用户提供的代价函数。代价函数可对接仿真器，组合任务误差、动作平滑和几何约束；通过固定种子复现采样。`containerloop.demo` 给出一个合成点质量模型示例。

`containerloop.policy.GrootAdapter` 接受 GR00T 的 `get_action` policy 实例及显式的观测、动作转换回调，输出带时间戳的 SE(3) 位姿和夹爪命令；模型加载、模态、动作坐标系和单位由用户配置，参考 [NVIDIA Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)。`MockPolicy` 仅用于接口测试。

`python -m containerloop.ablation --evaluator my_experiment:run --conditions conditions.json --output runs/ablation.jsonl` 使用同一批条件调用外部评测函数，传入 direct、smoothing、geometry、feedback、full 五组开关。外部评测函数负责实际仿真或实验并返回 success、grasp_drift_m、constraint_violations、execution_time_s、failure_reason；`evaluation.summarize_jsonl` 汇总运行记录。接口验证与实验日志的证据范围见 [简历证据索引](docs/RESUME_EVIDENCE.md)。

评测 CSV 字段为 `condition_id,method,success,drift_m`，成功标记为 `0` 或 `1`。每种方法在同一初始条件下使用相同 `condition_id`，每组只能有一条记录：

```python
from containerloop.evaluation import read_results, summarize, paired
rows = read_results('runs/evaluation.csv')
print(summarize(rows))
print(paired(rows, 'baseline', 'closed_loop'))
```

`summarize` 输出次数、成功数、成功率和平均位置漂移；`paired` 只比较两种方法共有的初始条件，并返回匹配数、成功数及平均漂移差。实验数据和模型接入时，应保留条件、配置和随机种子，便于重复评测。
