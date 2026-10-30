"""GR00T get_action adapter with explicit modality and robot-specific transforms."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .geometry import check_transform


@dataclass(frozen=True)
class Observation:
    rgb: Any
    depth: Any
    proprioception: Any
    instruction: str
    timestamp: float
    robot_config: dict


@dataclass(frozen=True)
class SE3Action:
    pose: np.ndarray
    gripper: float
    timestamp: float


class GrootAdapter:
    def __init__(self, policy, encode_observation, decode_actions):
        self.policy, self.encode_observation, self.decode_actions = policy, encode_observation, decode_actions

    def predict(self, observation):
        if not np.isfinite(observation.timestamp):
            raise ValueError('invalid observation timestamp')
        outputs, info = self.policy.get_action(self.encode_observation(observation))
        actions = list(self.decode_actions(outputs, observation))
        if not actions:
            raise ValueError('empty action chunk')
        for a in actions:
            check_transform(a.pose)
            if not np.isfinite([a.gripper, a.timestamp]).all():
                raise ValueError('invalid action')
        return actions, info


class MockPolicy:
    """Interface test double; not a GR00T checkpoint."""
    def get_action(self, observation):
        return {'action': np.zeros((1, 2, 7))}, {'mock': True}
