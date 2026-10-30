import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from containerloop.geometry import inverse, relative, world_target, pose_error, project_delta, rotation_vector
from containerloop.control import Stage, StageGate, damped_ik_step, joint_pd
from containerloop.data import nearest_indices, rigid_calibration
from containerloop.episodes import align_episode, write_jsonl
from containerloop.optim import cem
from containerloop.evaluation import read_results, summarize, paired


class GeometryTests(unittest.TestCase):
    def test_relative_pose_and_inverse(self):
        a, b = np.eye(4), np.eye(4)
        a[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        a[:3, 3], b[:3, 3] = [1, 2, 3], [2, 2, 3]
        np.testing.assert_allclose(a @ inverse(a), np.eye(4), atol=1e-9)
        np.testing.assert_allclose(world_target(a, relative(a, b)), b, atol=1e-9)
        np.testing.assert_allclose(pose_error(b, b), np.zeros(6), atol=1e-9)

    def test_rotation_near_pi_and_projection(self):
        R = np.diag([1., -1., -1.])
        self.assertAlmostEqual(np.linalg.norm(rotation_vector(R)), np.pi)
        delta = project_delta([1, 1, 0, 0, 0, 2], .2, .3, [0, 0, 0], [1, 1, 1], [.95, .5, .5])
        self.assertLessEqual(np.linalg.norm(delta[:3]), .2 + 1e-10)
        self.assertAlmostEqual(delta[0], .05)
        self.assertAlmostEqual(delta[5], .3)
        with self.assertRaises(ValueError):
            inverse(np.zeros((4, 4)))


class ControlTests(unittest.TestCase):
    def test_stage_transitions_and_invalid_observation(self):
        gate = StageGate()
        self.assertEqual(gate.update(.1, .1), Stage.APPROACH)
        self.assertEqual(gate.update(.01, .1), Stage.GRASP)
        self.assertEqual(gate.update(.01, .1, grasp_confirmed=True), Stage.TRANSPORT)
        self.assertEqual(gate.update(.01, .1), Stage.PLACE)
        self.assertEqual(gate.update(.01, .1, release_confirmed=True), Stage.DONE)
        with self.assertRaises(ValueError): gate.update(float('nan'), 0)

    def test_ik_bounds_collision_and_pd(self):
        J = np.eye(6)
        q, ok = damped_ik_step(J, np.ones(6), np.zeros(6), -np.ones(6), np.ones(6), max_step=.1)
        self.assertTrue(ok)
        np.testing.assert_allclose(q, .1 * np.ones(6))
        q, ok = damped_ik_step(J, np.ones(6), np.zeros(6), -np.ones(6), np.ones(6), collision_free=lambda _: False)
        self.assertFalse(ok)
        np.testing.assert_array_equal(q, np.zeros(6))
        np.testing.assert_allclose(joint_pd([0], [0.1], [1], 10, 1, 2), [2])


class DataTests(unittest.TestCase):
    def test_calibration_recovers_transform(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = source @ R.T + [1, 2, 3]
        T, rms = rigid_calibration(source, target)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-8)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-8)
        self.assertLess(rms, 1e-8)
        with self.assertRaises(ValueError): rigid_calibration(source[:2], target[:2])

    def test_alignment_and_jsonl(self):
        np.testing.assert_array_equal(nearest_indices([0, 1, 2], [.05, 1.02], .1), [0, 1, -1])
        rows = align_episode([{'timestamp': 0.0}, {'timestamp': 1.0}], {'front': [{'timestamp': .02, 'path': 'f.png'}]}, .1)
        self.assertEqual(rows[0]['cameras']['front']['path'], 'f.png')
        self.assertIsNone(rows[1]['cameras']['front'])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'episode.jsonl'
            write_jsonl(rows, p)
            self.assertEqual(len(p.read_text().splitlines()), 2)


class EvaluationTests(unittest.TestCase):
    def test_cem_improves_and_is_reproducible(self):
        def objective(x): return float(np.sum((x - .2)**2))
        result, cost = cem(objective, 3, 2, -1, 1, population=100, elites=10, iterations=8, seed=2)
        self.assertLess(cost, .02)
        np.testing.assert_array_equal(result, cem(objective, 3, 2, -1, 1, population=100, elites=10, iterations=8, seed=2)[0])

    def test_paired_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'results.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([['condition_id','method','success','drift_m'], ['1','a','0','.03'], ['1','b','1','.01'], ['2','a','1','.02']])
            rows = read_results(path)
            self.assertEqual(summarize(rows)['a']['success_rate'], .5)
            result = paired(rows, 'a', 'b')
            self.assertEqual(result['matched_conditions'], 1)
            self.assertAlmostEqual(result['mean_drift_difference_b_minus_a_m'], -.02)
