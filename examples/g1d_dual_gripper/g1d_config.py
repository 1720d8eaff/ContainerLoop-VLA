"""GR00T N1.7 custom embodiment: G1-D arms and two 1-DoF grippers.

Use with --embodiment-tag NEW_EMBODIMENT --modality-config-path PATH.
Dataset and policy actions are ABSOLUTE targets, not joint deltas.
Gripper values remain in the recording's native qpos command scale.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

# Order must agree with meta/modality.json and g1d_adapter.py.
KEYS = ["left_arm", "right_arm", "left_gripper", "right_gripper"]
ACTION_HORIZON = 16  # Design choice; train and deploy with the same configuration.

g1d_config = {
    # color_3 is the only view present in the inspected public demonstrations.
    "video": ModalityConfig(delta_indices=[0], modality_keys=["color_3"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=KEYS.copy()),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=KEYS.copy(),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1d_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
