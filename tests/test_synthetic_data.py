"""Data-driven checks of the repository's existing geometry and IK code."""
import csv
import json
from pathlib import Path
import numpy as np
from containerloop.geometry import project_delta, relative, world_target
from containerloop.control import damped_ik_step


def test_synthetic_geometry_and_action_constraints():
    with (Path(__file__).parent/'fixtures/synthetic_states.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)>1
    for i,r in enumerate(rows):
        assert r['source']=='synthetic_sdv' and r['validation_status']=='passed_kinematic_checks'
        obj=np.eye(4);obj[:3,3]=json.loads(r['object_position'])
        tool=np.eye(4);tool[:3,3]=json.loads(r['left_ee_position'])
        np.testing.assert_allclose(world_target(obj,relative(obj,tool)),tool,atol=1e-9)
        assert abs(np.linalg.norm(json.loads(r['left_ee_quaternion']))-1)<1e-6
        previous=json.loads(rows[max(i-1,0)]['left_ee_position'])
        action=np.r_[np.asarray(json.loads(r['left_ee_position']))-previous,[0.,0.,0.]]
        clipped=project_delta(action,.01,.01)
        assert np.linalg.norm(clipped[:3])<=.01000001 and np.linalg.norm(clipped[3:])<=.01000001
        proposed=np.asarray(json.loads(r['proposed_action']))
        projected=np.asarray(json.loads(r['projected_action']))
        np.testing.assert_allclose(proposed,projected)
        # Identity Jacobian exercises the public IK solver interface; FK/IK
        # checks with the G1-D model run in the data generator itself.
        q=np.asarray(json.loads(r['joint_position']))[:6]
        q_next,ok=damped_ik_step(np.eye(6),clipped,q,q-.05,q+.05,max_step=.01)
        assert ok and np.all(np.abs(q_next-q)<=.01000001)
