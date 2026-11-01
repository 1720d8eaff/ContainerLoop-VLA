# 用户提供的 GR00T 训练日志

[原始日志](training.log)由仓库所有者提供，原文件名为 `GR00T日志.log`；归档保持原始字节。文件记录的是训练与验证过程，没有真机观测、动作执行或任务成功率评测记录。

## 日志中记录的信息

- 基础模型标记：GR00T-N1.7；训练方式标记：LoRA。
- 数据记录：300 episodes、184,900 frames、4 tasks，图像键 head_rgb 和 wrist_left_rgb。
- 状态和动作维度：16；动作块长度：16；总训练步数：20,000。
- 文件末尾记录：best_validation_step=16000、best_validation_loss=0.610517。

以上是文件自身报告的内容。本次上传只归档文件，未独立复跑训练、加载对应 checkpoint 或核验设备运行情况。日志中的 `checkpoint_created`、`actual_training_performed` 等字段是日志内容，本目录没有附带实际模型权重。

## 与机器人适配配置的关系

本仓库的[双夹爪适配示例](../../../examples/g1d_dual_gripper/README.md)默认使用 color_3 单视角；本日志记录 head_rgb、wrist_left_rgb 双视角和 LoRA 训练。它们是不同配置，复现本日志所述训练时需使用对应的数据、模态配置、LoRA实现和启动参数，不能直接把单视角示例当作该次训练的原始配置。

## 文件校验

SHA-256：`23e0a1137a891793ff767ba834f67073ab694d4be0adeffc954ea939129424c4`

本次归档保持日志原样，未修改数值。
