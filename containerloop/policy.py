"""GR00T get_action adapter with explicit modality and robot-specific transforms."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .geometry import check_transform


def decode_chunk(chunk,mean,std,eef,timestamp,control_dt,mode,translation_limit,rotation_limit):
    """Decode standardized [xyz, rotation-vector, gripper] rows.

    Relative increments use world axes and accumulate over the chunk. Absolute
    rows are world poses. This explicit 7D convention is not universal GR00T
    output; NVIDIA get_action already returns physical units, so pass mean=0,
    std=1 only when its action modality uses this exact representation.
    """
    from scipy.spatial.transform import Rotation
    from .geometry import project_delta,pose_error
    chunk=np.asarray(chunk,float);mean=np.asarray(mean,float);std=np.asarray(std,float)
    if chunk.ndim!=2 or chunk.shape[1]!=7 or len(chunk)==0 or mean.shape!=(7,) or std.shape!=(7,) or np.any(std<=0) or not all(np.isfinite(x).all() for x in (chunk,mean,std)) or not np.isfinite([timestamp,control_dt]).all() or control_dt<=0 or mode not in ('relative','absolute'):raise ValueError('invalid action convention/statistics')
    current=check_transform(eef).copy();result=[]
    for i,row in enumerate(chunk*std+mean):
        if mode=='relative':delta=row[:6]
        else:
            target=np.eye(4);target[:3,3]=row[:3];target[:3,:3]=Rotation.from_rotvec(row[3:6]).as_matrix();delta=pose_error(current,target)
        delta=project_delta(delta,translation_limit,rotation_limit)
        current[:3,3]+=delta[:3];current[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()@current[:3,:3]
        result.append(SE3Action(current.copy(),float(np.clip(row[6],0,1)),float(timestamp+i*control_dt)))
    return result


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
