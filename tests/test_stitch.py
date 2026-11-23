import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from calibration import Calibration, board_points, make_ground_maps, solve_extrinsics
from stitch import camera_weights, check_pair, compose, match_gains


class BlendTests(unittest.TestCase):
    def test_weights_zero_outside_and_one_inside(self):
        mapx = np.array([[-1, 5, 100, 319, 400]], np.float32)
        mapy = np.full((1, 5), 100, np.float32)
        weights = camera_weights(mapx, mapy, (320, 240), feather=40)
        self.assertEqual(weights[0, 0], 0)
        self.assertEqual(weights[0, 4], 0)
        self.assertEqual(weights[0, 3], 0)
        self.assertAlmostEqual(float(weights[0, 1]), 5 / 40, places=5)
        self.assertEqual(weights[0, 2], 1)

    def test_compose_blends_and_leaves_gaps_black(self):
        shape = (1, 4)
        a = np.full((*shape, 3), 100, np.float32)
        b = np.full((*shape, 3), 200, np.float32)
        wa = np.array([[1, 1, 0.5, 0]], np.float32)
        wb = np.array([[0, 0, 0.5, 0]], np.float32)
        out = compose([a, b], [wa, wb])
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[0, 0, 0], 100)
        self.assertEqual(out[0, 2, 0], 150)
        self.assertEqual(out[0, 3, 0], 0)

    def test_gain_matching_evens_out_overlap(self):
        full = np.ones((40, 40), np.float32)
        a = np.full((40, 40, 3), 100, np.float32)
        b = np.full((40, 40, 3), 200, np.float32)
        gains = match_gains([a, b], [full, full])
        self.assertAlmostEqual(float(gains[0] * 100), float(gains[1] * 200), delta=1)
        self.assertAlmostEqual(float(np.prod(gains)), 1.0, places=4)

    def test_compose_requires_input(self):
        with self.assertRaises(ValueError):
            compose([], [])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            (640, 480), (9, 6), np.array([[300., 0, 320], [0, 300., 240], [0, 0, 1]]),
            np.zeros((4, 1)), 0.3,
        )
        pitch = np.radians(40)
        rot = np.array([[1, 0, 0], [0, -np.sin(pitch), -np.cos(pitch)], [0, np.cos(pitch), -np.sin(pitch)]])
        self.rvec = cv2.Rodrigues(rot)[0]
        self.tvec = -rot @ np.array([[0.0], [-1500.0], [1500.0]])
        args = ((9, 6), 100.0, (-400.0, 1000.0), 0.0)
        points, _ = cv2.fisheye.projectPoints(
            board_points(*args), self.rvec, self.tvec, self.calibration.matrix, self.calibration.distortion
        )
        self.extrinsics = solve_extrinsics(points, self.calibration, *args[1:])

    def test_floor_texture_lands_in_ground_view(self):
        maps = make_ground_maps(self.calibration, self.extrinsics, 0.1, (200, 200))
        weights = camera_weights(*maps, self.calibration.size)
        self.assertGreater(weights.max(), 0.99)
        self.assertEqual(weights.shape, (200, 200))
        frame = np.full((480, 640, 3), 90, np.uint8)
        warped = cv2.remap(frame, *maps, cv2.INTER_LINEAR).astype(np.float32)
        out = compose([warped], [weights])
        covered = weights > 0.99
        self.assertTrue((out[covered] == 90).all())
        self.assertTrue((out[weights == 0] == 0).all())

    def test_check_pair_rejects_mismatch(self):
        check_pair(self.calibration, self.extrinsics)
        other = Calibration((800, 600), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1)
        with self.assertRaises(ValueError):
            check_pair(other, self.extrinsics)


if __name__ == "__main__":
    unittest.main()
