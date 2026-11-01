"""Pure data adapter. No DDS/ROS publishers, serial writes, or robot motion.

Dependencies: numpy. Camera input must already be synchronized RGB uint8.
"""
import numpy as np

STREAMS = (
    ("left_arm", "left_arm", 7),
    ("right_arm", "right_arm", 7),
    ("left_gripper", "left_ee", 1),
    ("right_gripper", "right_ee", 1),
)
LEFT_ARM_MOTOR_IDS = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_MOTOR_IDS = (22, 23, 24, 25, 26, 27, 28)
JOINT_SUFFIXES = (
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
)


def _vector(value, dim, name):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (dim,) or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite vector of shape ({dim},), got {a.shape}")
    return a


def pack_qpos(qpos):
    """Pack recorder keys left_arm/right_arm/left_ee/right_ee into 16 values.

    Pass states_qpos for observation; pass actions_qpos for training targets.
    Missing channels are rejected. No zero-filled or mirrored observations.
    """
    return np.concatenate([
        _vector(qpos[source], dim, source) for _, source, dim in STREAMS
    ])


def unpack_vector(vector):
    """Split the 16-D absolute state/action contract into named streams."""
    a = _vector(vector, 16, "vector")
    offset, out = 0, {}
    for key, _, dim in STREAMS:
        out[key] = a[offset:offset + dim].copy()
        offset += dim
    return out


def qpos_from_lowstate(motor_q, left_gripper_q, right_gripper_q):
    """Read arm positions from an existing 29/35-slot low-state q array.

    Gripper feedback must be supplied separately by the actual gripper reader.
    This function does not assume that grippers occupy lowstate slots 31/33.
    The two gripper inputs must be one-element arrays, not trigger values.
    """
    q = np.asarray(motor_q, dtype=np.float32)
    if q.ndim != 1 or len(q) < 29:
        raise ValueError("motor_q must be a one-dimensional array with at least 29 slots")
    out = {
        "left_arm": q[list(LEFT_ARM_MOTOR_IDS)],
        "right_arm": q[list(RIGHT_ARM_MOTOR_IDS)],
        "left_ee": left_gripper_q,
        "right_ee": right_gripper_q,
    }
    pack_qpos(out)  # Check only selected channels; unused motor slots are not model inputs.
    return out


def make_observation(rgb, states_qpos, instruction):
    """Construct the official Gr00tPolicy observation schema, B=1 and T=1.

    No resize, BGR conversion, clock alignment or value normalization is hidden.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an RGB uint8 numpy array")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("rgb must have shape [height, width, 3]")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    streams = unpack_vector(pack_qpos(states_qpos))
    return {
        "video": {"color_3": rgb[None, None].copy()},
        "state": {key: a[None, None] for key, a in streams.items()},
        "language": {"task": [[instruction]]},
    }


def decode_action_chunk(actions):
    """Convert get_action output to [horizon, 16] absolute native targets.

    Official get_action returns unnormalized actions. Do not unnormalize again,
    add current joints, convert gripper q to width, or apply a guessed offset.
    This is data decoding; a controller still needs limits and timing checks.
    """
    expected = {key for key, _, _ in STREAMS}
    if set(actions) != expected:
        raise ValueError(f"action keys must be exactly {sorted(expected)}")
    chunks, horizon = [], None
    for key, _, dim in STREAMS:
        a = np.asarray(actions[key], dtype=np.float32)
        if a.ndim != 3 or a.shape[0] != 1 or a.shape[1] < 1 or a.shape[2] != dim:
            raise ValueError(f"{key}: expected [1, horizon, {dim}], got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{key}: nonfinite action")
        if horizon is not None and horizon != a.shape[1]:
            raise ValueError("action streams have different horizons")
        horizon = a.shape[1]
        chunks.append(a[0])
    return np.concatenate(chunks, axis=-1)


def recorder_targets(vector):
    """Return one absolute action in the recorder's original key naming."""
    streams = unpack_vector(vector)
    return {source: streams[key].tolist() for key, source, _ in STREAMS}
