"""Rigid transforms and object-relative pose corrections (metres, radians)."""
import numpy as np


def check_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("invalid rigid transform")
    return T


def inverse(T):
    T = check_transform(T)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def relative(object_world, tool_world):
    """Return T_object_tool from world-frame object and tool poses."""
    return inverse(object_world) @ check_transform(tool_world)


def world_target(object_world, object_tool_target):
    return check_transform(object_world) @ check_transform(object_tool_target)


def rotation_vector(R):
    """Log map SO(3), including the near-pi case."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError("invalid rotation")
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return skew / 2
    if np.pi - theta < 1e-5:
        values, vectors = np.linalg.eigh((R + np.eye(3)) / 2)
        axis = vectors[:, np.argmax(values)]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis * theta
    return skew * theta / (2 * np.sin(theta))


def pose_error(current, target):
    """World-frame translation and rotation error, suitable for a world-frame Jacobian."""
    current, target = check_transform(current), check_transform(target)
    return np.r_[target[:3, 3] - current[:3, 3], rotation_vector(target[:3, :3] @ current[:3, :3].T)]


def project_delta(delta, translation_limit, rotation_limit, lower_xyz=None, upper_xyz=None, current_xyz=None):
    """Clip Cartesian step norms and optionally project the resulting position to a box."""
    delta = np.asarray(delta, dtype=float).copy()
    if delta.shape != (6,) or not np.isfinite(delta).all() or not np.isfinite([translation_limit, rotation_limit]).all() or min(translation_limit, rotation_limit) < 0:
        raise ValueError("invalid delta or limits")
    for part, limit in ((slice(0, 3), translation_limit), (slice(3, 6), rotation_limit)):
        norm = np.linalg.norm(delta[part])
        if norm > limit and norm > 0:
            delta[part] *= limit / norm
    if any(x is not None for x in (lower_xyz, upper_xyz, current_xyz)):
        if any(x is None for x in (lower_xyz, upper_xyz, current_xyz)):
            raise ValueError("workspace projection needs position and both bounds")
        lower, upper, position = (np.asarray(x, dtype=float) for x in (lower_xyz, upper_xyz, current_xyz))
        if any(x.shape != (3,) or not np.isfinite(x).all() for x in (lower, upper, position)) or np.any(lower > upper):
            raise ValueError("invalid workspace")
        if np.any(position < lower) or np.any(position > upper):
            raise ValueError("current position outside workspace")
        delta[:3] = np.clip(position + delta[:3], lower, upper) - position
    return delta
