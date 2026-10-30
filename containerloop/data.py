"""Timestamp synchronization and rigid extrinsic calibration."""
import numpy as np


def nearest_indices(reference_times, sensor_times, max_offset):
    """Return nearest sensor index per reference time, or -1 outside tolerance."""
    ref, sensor = np.asarray(reference_times, float), np.asarray(sensor_times, float)
    if ref.ndim != 1 or sensor.ndim != 1 or not np.isfinite(ref).all() or not np.isfinite(sensor).all() or not np.isfinite(max_offset) or max_offset < 0 or np.any(np.diff(ref) < 0) or np.any(np.diff(sensor) < 0):
        raise ValueError("times must be finite, sorted vectors and tolerance nonnegative")
    if sensor.size == 0:
        return np.full(ref.shape, -1, dtype=int)
    right = np.clip(np.searchsorted(sensor, ref), 0, sensor.size - 1)
    left = np.maximum(right - 1, 0)
    idx = np.where(np.abs(sensor[left] - ref) <= np.abs(sensor[right] - ref), left, right)
    return np.where(np.abs(sensor[idx] - ref) <= max_offset, idx, -1)


def rigid_calibration(source_points, target_points):
    """Fit T_target_source from matched 3D points via Kabsch; return transform and RMS residual."""
    src, dst = np.asarray(source_points, float), np.asarray(target_points, float)
    if src.ndim != 2 or src.shape[1:] != (3,) or src.shape != dst.shape or len(src) < 3 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("expected at least three finite 3D point pairs")
    a, b = src - src.mean(axis=0), dst - dst.mean(axis=0)
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        raise ValueError("calibration points must not be collinear")
    U, _, Vt = np.linalg.svd(a.T @ b)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ fix @ U.T
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    rms = np.sqrt(np.mean(np.sum((src @ R.T + t - dst)**2, axis=1)))
    return T, float(rms)
