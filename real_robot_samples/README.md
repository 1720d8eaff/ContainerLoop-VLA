# 真实机器人容器操作：公开展示子集

画面展示机器人双臂与容器操作

## 文件与评测入口

- [样例压缩包](container_real_samples.zip)：公开 episode_0002、episode_0007、episode_0009 的抽样图像与 `trajectory.json`。
- [数据清单](manifest.json)：列出各段原始帧数、保留帧数，并声明源数据有 300 段；公开压缩包为其中 3 段抽样。
- [真机评测逐回合记录](../artifacts/contact/real_robot_episodes.jsonl)：`rr01` 至 `rr30` 的匹配条件记录，两种方法各 30 回合；`vla_closed_loop` 成功 21 回合，`rule_control` 成功 17 回合。
- [评测汇总](../artifacts/contact/real_robot_evaluation.json)：包含成功率、配对结果（双方成功 15、双方失败 7、仅 VLA 成功 6、仅规则成功 2）及抓取放置的成功判定。

这些文件已位于 main 主干。示教样例没有逐条关联到上述评测回合；日志中的策略名称未关联到可核对的微调 checkpoint。示教、评测和倒球仿真的证据范围应分别按对应文件说明理解。

## 数据处理

- 原始图像元信息标注 30 FPS；每隔 6 帧抽取一帧并保留末帧.
- 只公开 color_3 视角，裁掉原图上方 96 像素后缩放，移除图像元数据
- 每个样例保留抽取帧对应的左右臂和夹爪状态、动作 qpos；

详见各回合的 trajectory.json。仅作真实采集过程和数据结构展示。
