import os
from pathlib import Path

import numpy as np
import pytest

from containerloop.policy import Observation, decode_chunk

FIXTURE = Path(__file__).parent / "fixtures" / "groot_chunk.npz"
CKPT = Path(os.environ.get("GROOT_CKPT", "checkpoints/groot_n1.7_container"))

def test_decode_matches_recorded_chunk():
    data = np.load(FIXTURE)
    actions = decode_chunk(
        data["action"],
        data["mean"],
        data["std"],
        data["eef"],
        float(data["timestamp"]),
        0.033,
        "relative",
        0.02,
        0.1,
    )
    assert len(actions) == data["action"].shape[0]
    assert all(0.0 <= item.gripper <= 1.0 for item in actions)
    assert all(np.isfinite(item.pose).all() for item in actions)
    np.testing.assert_allclose(actions[0].pose[:3, 3], data["first_position"], atol=1e-6)
    np.testing.assert_allclose(actions[0].pose[:3, :3], data["first_rotation"], atol=1e-6)

def test_decode_rejects_bad_statistics():
    data = np.load(FIXTURE)
    with pytest.raises(ValueError):
        decode_chunk(data["action"], data["mean"], data["std"][:-1], data["eef"], 0.0, 0.033, "relative", 0.02, 0.1)

def test_decode_rejects_unknown_mode():
    data = np.load(FIXTURE)
    with pytest.raises(ValueError):
        decode_chunk(data["action"], data["mean"], data["std"], data["eef"], 0.0, 0.033, "delta_abs", 0.02, 0.1)

@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not available")
def test_policy_smoke():
    from containerloop.policy import GrootPolicy

    policy = GrootPolicy(CKPT, horizon=16, control_dt=0.033)
    observation = Observation(
        images={"front": np.zeros((480, 640, 3), dtype=np.uint8), "wrist": np.zeros((480, 640, 3), dtype=np.uint8)},
        depth={},
        proprioception=np.zeros(14, dtype=np.float32),
        instruction="pick up the beaker and place it on the stand",
        timestamp=10.0,
        robot_config={"eef_pose": np.eye(4)},
    )
    actions = policy.act(observation)
    assert len(actions) == 16
    assert all(0.0 <= item.gripper <= 1.0 for item in actions)
    assert all(np.linalg.norm(item.pose[:3, 3] - observation.robot_config["eef_pose"][:3, 3]) < 0.5 for item in actions)
    policy.close()
