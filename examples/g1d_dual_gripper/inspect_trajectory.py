"""Inspect a real recorder trajectory.json and optionally export adapter arrays.

An NPZ produced here is an intermediate, NOT a complete LeRobot dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from g1d_adapter import pack_qpos


def inspect(path):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    frames = data["frames"]
    if not frames:
        raise ValueError("trajectory has no frames")
    states = np.stack([pack_qpos(f["states_qpos"]) for f in frames])
    actions = np.stack([pack_qpos(f["actions_qpos"]) for f in frames])
    names = [f["image"] for f in frames]
    for name in names:
        image = (path.parent / name).resolve()
        if path.parent.resolve() not in image.parents or not image.is_file():
            raise ValueError(f"image missing or outside episode directory: {name}")
    indices = np.array([f["source_frame_index"] for f in frames], dtype=np.int64)
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("source frame indices must be nonnegative and strictly increasing")
    # Retain provenance: the public subset contains only NOMINAL timestamps.
    times = np.array([f["relative_time_seconds_nominal"] for f in frames], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("nominal timestamps must be finite and strictly increasing")
    summary = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_episode": data.get("source_episode"),
        "frames": len(frames), "state_dim": 16, "action_dim": 16,
        "camera": data.get("camera"),
        "timestamp_kind": "nominal_from_source_fps_not_measured_sync",
        "source_fps_nominal": data.get("source_fps_nominal"),
        "retained_source_frame_gaps": sorted(set(np.diff(indices).tolist())),
        "state_min": states.min(0).tolist(), "state_max": states.max(0).tolist(),
        "action_min": actions.min(0).tolist(), "action_max": actions.max(0).tolist(),
    }
    return summary, states, actions, times, indices, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectory", type=Path)
    p.add_argument("--output", type=Path, help="optional new NPZ file; no overwrite")
    args = p.parse_args()
    summary, states, actions, times, indices, names = inspect(args.trajectory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, states=states, actions=actions,
                                timestamps_nominal=times, source_frame_indices=indices,
                                image_paths=np.array(names),
                                provenance_json=np.array(json.dumps(summary, ensure_ascii=False)))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
