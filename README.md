# ContainerLoop-VLA

面向宇树 G1-D 容器操作的物体中心几何控制、数据对齐与评测工具。代码将目标位姿表示在物体坐标系中，组合动作投影、阶段门控、反馈纠偏、IK 和关节 PD 计算，并提供相机外参拟合、时间戳同步及 CEM 动作优化接口。

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

`tests/fixtures/synthetic_states.csv` 是明确标记为 `synthetic_sdv` 的少量合成测试样本；pytest 使用它调用几何投影与 IK 接口，不是机器人实验记录。

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

评测 CSV 字段为 `condition_id,method,success,drift_m`，成功标记为 `0` 或 `1`。每种方法在同一初始条件下使用相同 `condition_id`，每组只能有一条记录：

```python
from containerloop.evaluation import read_results, summarize, paired
rows = read_results('runs/evaluation.csv')
print(summarize(rows))
print(paired(rows, 'baseline', 'closed_loop'))
```

`summarize` 输出次数、成功数、成功率和平均位置漂移；`paired` 只比较两种方法共有的初始条件，并返回匹配数、成功数及平均漂移差。实验数据和模型接入时，应保留条件、配置和随机种子，便于重复评测。
