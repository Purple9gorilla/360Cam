import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    Calibration, board_points, load_extrinsics, make_ground_maps, rotated_180, save_extrinsics, solve_extrinsics,
)


class ExtrinsicsTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Calibration(
            (1280, 720), (9, 6), np.array([[600., 0, 640], [0, 600., 360], [0, 0, 1]]),
            np.array([[-0.05], [0.01], [0.0], [0.0]]), 0.3, "device-one",
        )
        # Camera 1.2 m above the floor, 1 m behind the board, looking forward and down.
        pitch = np.radians(30)
        rot = np.array([
            [1, 0, 0],
            [0, -np.sin(pitch), -np.cos(pitch)],
            [0, np.cos(pitch), -np.sin(pitch)],
        ])
        self.rvec = cv2.Rodrigues(rot)[0]
        self.tvec = -rot @ np.array([[0.0], [-1000.0], [1200.0]])
        self.args = ((9, 6), 100.0, (-400.0, 800.0), 0.0)

    def project(self):
        obj = board_points(*self.args)
        points, _ = cv2.fisheye.projectPoints(
            obj, self.rvec, self.tvec, self.calibration.matrix, self.calibration.distortion
        )
        return points

    def test_recovers_known_pose(self):
        result = solve_extrinsics(self.project(), self.calibration, *self.args[1:], "device-one", "")
        np.testing.assert_allclose(result.rotation(), cv2.Rodrigues(self.rvec)[0], atol=1e-3)
        np.testing.assert_allclose(result.position_mm(), [0, -1000, 1200], atol=2)
        self.assertLess(result.rms, 0.1)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            solve_extrinsics(np.zeros((3, 1, 2)), self.calibration, *self.args[1:])
        with self.assertRaises(ValueError):
            solve_extrinsics(self.project(), self.calibration, 0, (0, 0), 0)

    def test_round_trip(self):
        result = solve_extrinsics(self.project(), self.calibration, *self.args[1:], "device-one", "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extrinsics.json"
            save_extrinsics(path, result)
            loaded = load_extrinsics(path)
            np.testing.assert_allclose(loaded.rvec, result.rvec)
            np.testing.assert_allclose(loaded.tvec, result.tvec)
            self.assertEqual(loaded.origin_mm, result.origin_mm)
            self.assertEqual(loaded.device_path, "device-one")
            path.write_text('{"version": 2}')
            with self.assertRaises(ValueError):
                load_extrinsics(path)

    def test_mirrored_board_is_rejected_below_floor(self):
        obj = board_points(*self.args) * [1, -1, 1]
        points, _ = cv2.fisheye.projectPoints(
            obj, self.rvec, self.tvec, self.calibration.matrix, self.calibration.distortion
        )
        with self.assertRaises(ValueError):
            solve_extrinsics(points, self.calibration, *self.args[1:])

    def test_upside_down_camera_uses_flipped_intrinsics(self):
        flipped = rotated_180(self.calibration)
        points, _ = cv2.fisheye.projectPoints(
            board_points(*self.args), self.rvec, self.tvec, flipped.matrix, flipped.distortion
        )
        result = solve_extrinsics(points, self.calibration, *self.args[1:], rotated=True)
        self.assertTrue(result.rotated)
        np.testing.assert_allclose(result.position_mm(), [0, -1000, 1200], atol=2)

    def test_ground_maps_shape_and_visibility(self):
        result = solve_extrinsics(self.project(), self.calibration, *self.args[1:])
        mapx, mapy = make_ground_maps(self.calibration, result, scale=0.1, canvas=(200, 200))
        self.assertEqual(mapx.shape, (200, 200))
        self.assertEqual(mapx.dtype, np.float32)
        # The board origin must map back onto the detected first corner.
        u, v = int(round(100 + -400 * 0.1)), int(round(100 - 800 * 0.1))
        first = self.project()[0, 0]
        self.assertAlmostEqual(float(mapx[v, u]), float(first[0]), delta=12)
        self.assertAlmostEqual(float(mapy[v, u]), float(first[1]), delta=12)


if __name__ == "__main__":
    unittest.main()
