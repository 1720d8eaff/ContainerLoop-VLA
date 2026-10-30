# 简历技术证据索引

| 技术点 | 代码 | 测试 | 日志和验证范围 |
| --- | --- | --- | --- |
| GR00T N1.7 | `containerloop/policy.py` | `tests/test_adapters.py` | 接口使用 MockPolicy；真实模型、权重与运行日志需另行提供 |
| SE(3)、阶段门控、IK、PD | `containerloop/geometry.py`, `control.py` | `tests/test_core.py` | 离线数值及碰撞回调测试 |
| 数据对齐 | `containerloop/data.py`, `episodes.py` | `tests/test_core.py` | 离线标定和时间对齐测试 |
| CEM 与消融 | `containerloop/optim.py`, `ablation.py`, `evaluation.py` | `tests/test_core.py`, `test_adapters.py` | 合成示例和外部评测器接口；192 次仿真实验的原始日志需另行提供 |
| 真机评测 | `containerloop/evaluation.py` | `tests/test_core.py` | 统计接口支持导入记录；30 组匹配真机实验日志需另行提供 |
