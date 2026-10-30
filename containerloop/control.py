"""Stage gating, damped least-squares IK and joint-space PD."""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Stage(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    PLACE = "place"
    DONE = "done"


@dataclass
class StageGate:
    position_tolerance: float = 0.02
    angle_tolerance: float = 0.15
    stage: Stage = Stage.APPROACH

    def __post_init__(self):
        if not np.isfinite([self.position_tolerance, self.angle_tolerance]).all() or min(self.position_tolerance, self.angle_tolerance) < 0:
            raise ValueError("invalid tolerances")

    def update(self, position_error, angle_error, grasp_confirmed=False, release_confirmed=False):
        if not np.isfinite([position_error, angle_error]).all() or min(position_error, angle_error) < 0:
            raise ValueError("invalid errors")
        aligned = position_error <= self.position_tolerance and angle_error <= self.angle_tolerance
        if self.stage == Stage.APPROACH and aligned:
            self.stage = Stage.GRASP
        elif self.stage == Stage.GRASP and grasp_confirmed:
            self.stage = Stage.TRANSPORT
        elif self.stage == Stage.TRANSPORT and aligned:
            self.stage = Stage.PLACE
        elif self.stage == Stage.PLACE and release_confirmed:
            self.stage = Stage.DONE
        return self.stage


def damped_ik_step(jacobian, error, q, lower, upper, *, damping=0.05, max_step=0.1, collision_free=None):
    """Compute one bounded IK step; Jacobian and error must use the same frame."""
    J, error, q, lower, upper = (np.asarray(x, dtype=float) for x in (jacobian, error, q, lower, upper))
    if q.ndim != 1 or J.shape != (6, q.size) or error.shape != (6,) or lower.shape != q.shape or upper.shape != q.shape or not all(np.isfinite(x).all() for x in (J, error, q, lower, upper)) or np.any(lower > upper) or np.any(q < lower) or np.any(q > upper) or not np.isfinite([damping, max_step]).all() or damping <= 0 or max_step <= 0:
        raise ValueError("invalid IK input")
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), error)
    dq = np.clip(dq, -max_step, max_step)
    candidate = np.clip(q + dq, lower, upper)
    if collision_free is not None and not collision_free(candidate):
        return q.copy(), False
    return candidate, True


def joint_pd(q, dq, target, kp, kd, torque_limit):
    q, dq, target, kp, kd, limit = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (q, dq, target, kp, kd, torque_limit)])
    if not all(np.isfinite(x).all() for x in (q, dq, target, kp, kd, limit)) or np.any(kp < 0) or np.any(kd < 0) or np.any(limit < 0):
        raise ValueError("invalid PD input")
    return np.clip(kp * (target - q) - kd * dq, -limit, limit)
