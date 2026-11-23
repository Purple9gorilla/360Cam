import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration import Calibration, calibrate, find_corners, load_calibration, make_maps, save_calibration
from camera_devices import Camera, attach_hardware_ids, instance_from_path


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.value = Calibration(
            (640, 480), (9, 6), np.array([[300., 0, 320], [0, 300., 240], [0, 0, 1]]),
            np.array([[-0.1], [0.01], [0.0], [0.0]]), 0.3,
            r"\\?\usb#vid_1234&pid_5678#device#{guid}", r"usb\vid_1234&pid_5678\device"
        )

    def test_round_trip_and_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            save_calibration(path, self.value)
            result = load_calibration(path)
            self.assertEqual(result.size, self.value.size)
            self.assertEqual(result.pattern, self.value.pattern)
            self.assertEqual(result.device_path, self.value.device_path)
            np.testing.assert_array_equal(result.matrix, self.value.matrix)
            np.testing.assert_array_equal(result.distortion, self.value.distortion)
            maps = make_maps(result)
            self.assertEqual(maps[0].shape[:2], (480, 640))
            self.assertEqual(cv2.remap(np.zeros((480, 640, 3), np.uint8), *maps,
                                       interpolation=cv2.INTER_LINEAR).shape, (480, 640, 3))

    def test_invalid_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            save_calibration(path, self.value)
            valid = json.loads(path.read_text())
            for key, bad in (("version", 2), ("image_size", [0, 480]),
                             ("camera_matrix", [[1]]), ("distortion", [[float("nan")]]),
                             ("rms", -1), ("pattern", [1, 6])):
                data = dict(valid)
                data[key] = bad
                path.write_text(json.dumps(data))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_calibration(path)

    def test_reject_insufficient_views(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            calibrate([], (9, 6), (640, 480))

    def test_checkerboard_detection(self):
        image = np.full((9 * 40, 12 * 40, 3), 255, np.uint8)
        for y in range(7):
            for x in range(10):
                if (x + y) % 2 == 0:
                    image[(y + 1) * 40:(y + 2) * 40, (x + 1) * 40:(x + 2) * 40] = 0
        corners = find_corners(image, (9, 6))
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (54, 1, 2))
        self.assertIsNone(find_corners(np.zeros_like(image), (9, 6)))

    def test_synthetic_fisheye_calibration(self):
        pattern = (9, 6)
        obj = np.zeros((54, 1, 3), np.float64)
        obj[:, 0, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        samples = []
        rng = np.random.default_rng(42)
        for _ in range(18):
            rotation = rng.uniform(-0.4, 0.4, (3, 1))
            translation = np.array([[rng.uniform(-5, -2)], [rng.uniform(-3, -1)], [rng.uniform(12, 20)]])
            points, _ = cv2.fisheye.projectPoints(obj, rotation, translation,
                                                  self.value.matrix, self.value.distortion)
            samples.append(points)
        result = calibrate(samples, pattern, self.value.size)
        self.assertLess(result.rms, 0.01)
        np.testing.assert_allclose(result.matrix, self.value.matrix, atol=0.1)


class DeviceTests(unittest.TestCase):
    def test_parse_only_standard_interface_paths(self):
        path = r"\\?\usb#vid_1234&pid_5678#device#{guid}"
        self.assertEqual(instance_from_path(path), r"usb\vid_1234&pid_5678\device")
        self.assertEqual(instance_from_path("virtual camera"), "")
        self.assertEqual(instance_from_path(r"\\?\usb#missing"), "")

    def test_unknown_hardware_label(self):
        self.assertIn("hardware ID unavailable", Camera(0, "Camera").label)
        self.assertIn("USB\\VID_1234", Camera(1, "Camera", hardware_ids=(r"USB\VID_1234",)).label)

    def test_exact_instance_mapping_only(self):
        cameras = [Camera(0, "Camera", instance_id=r"usb\vid_1234\one"),
                   Camera(1, "Camera", instance_id=r"usb\vid_1234\two"),
                   Camera(2, "Camera")]
        entries = [{"PNPDeviceID": r"USB\VID_1234\ONE", "HardwareID": [r"USB\VID_1234"]},
                   {"PNPDeviceID": r"USB\VID_1234\OTHER", "HardwareID": ["OTHER"]}]
        result = attach_hardware_ids(cameras, entries)
        self.assertEqual(result[0].hardware_ids, (r"USB\VID_1234",))
        self.assertEqual(result[1].hardware_ids, ())
        self.assertEqual(result[2].hardware_ids, ())


if __name__ == "__main__":
    unittest.main()
