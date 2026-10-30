import tempfile
import unittest
from pathlib import Path
import numpy as np
from containerloop.policy import GrootAdapter, MockPolicy, Observation, SE3Action
from containerloop.ablation import run
from containerloop.geometry import object_frame, grasp_consistency, feedback_correction


class AdapterTests(unittest.TestCase):
    def test_object_frame_and_feedback(self):
        frame = object_frame([1, 2, 3], [1, 0, 0], [0, 0, 1])
        np.testing.assert_allclose(frame[:3, 3], [1, 2, 3])
        self.assertTrue(grasp_consistency(np.eye(4), np.eye(4), .01, .1))
        target = np.eye(4)
        target[0, 3] = 1
        self.assertAlmostEqual(feedback_correction(np.eye(4), target, 1., .1, .2)[0], .1)
        with self.assertRaises(ValueError):
            object_frame([0, 0, 0], [1, 0, 0], [1, 0, 0])

    def test_groot_adapter(self):
        obs = Observation(None, None, [0], 'move', 1., {})
        decoder = lambda output, o: [SE3Action(np.eye(4), .5, o.timestamp)]
        actions, info = GrootAdapter(MockPolicy(), lambda x: {}, decoder).predict(obs)
        self.assertEqual(len(actions), 1)
        self.assertTrue(info['mock'])
        self.assertEqual(actions[0].timestamp, 1.)

    def test_ablation_passes_distinct_flags(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            run(lambda condition, flags: seen.append(flags) or {'success': True}, ['1'], Path(d)/'run.jsonl')
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(seen[-1].values()))
        self.assertFalse(any(seen[0].values()))
