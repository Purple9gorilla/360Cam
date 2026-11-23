import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from calibration import Calibration
from camera_devices import Camera
from main import Application, CameraWorker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = Application.__new__(Application)
        self.app.generation = 4
        self.app.token = 7
        self.app.samples = []
        self.app.busy = True
        self.app.status = Mock()
        self.app.sample_status = Mock()
        self.app.result = Mock()
        self.app.camera = Camera(0, "Camera", device_path="device-one")
        self.app.frame_size = (640, 480)
        self.app.calibration = None
        self.app.maps = None
        self.app.columns = Mock()
        self.app.rows = Mock()
        self.app.ground_view = Mock()
        self.app.extrinsics_info = Mock()
        self.app.extrinsics = None
        self.app.ground_maps = None

    def test_stale_capture_ignored(self):
        self.app.handle_event(("capture", 3, np.zeros((54, 1, 2)), None))
        self.assertEqual(self.app.samples, [])
        self.assertTrue(self.app.busy)

    def test_capture_is_added_only_when_detected(self):
        self.app.handle_event(("capture", 4, None, None))
        self.assertEqual(len(self.app.samples), 0)
        self.app.handle_event(("capture", 4, np.zeros((54, 1, 2)), None))
        self.assertEqual(len(self.app.samples), 1)

    def test_load_mismatched_device_declined(self):
        value = Calibration((640, 480), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1, "device-two")
        with patch("main.filedialog.askopenfilename", return_value="file.json"), \
             patch("main.load_calibration", return_value=value), \
             patch("main.messagebox.askyesno", return_value=False) as ask, \
             patch("main.make_maps") as maps:
            self.app.load()
        ask.assert_called_once()
        maps.assert_not_called()
        self.assertIsNone(self.app.calibration)

    def test_load_mismatched_device_accepted(self):
        value = Calibration((640, 480), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1, "device-two")
        with patch("main.filedialog.askopenfilename", return_value="file.json"), \
             patch("main.load_calibration", return_value=value), \
             patch("main.messagebox.askyesno", return_value=True), \
             patch("main.make_maps", return_value=("mx", "my")):
            self.app.load()
        self.assertIs(self.app.calibration, value)

    def test_load_rejects_mismatched_resolution(self):
        value = Calibration((800, 600), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1, "device-one")
        with patch("main.filedialog.askopenfilename", return_value="file.json"), \
             patch("main.load_calibration", return_value=value), \
             patch("main.messagebox.showerror") as error, \
             patch("main.make_maps") as maps:
            self.app.load()
        error.assert_called_once()
        self.assertIn("resolution", error.call_args.args[1])
        maps.assert_not_called()

    def test_unknown_identity_requires_confirmation(self):
        value = Calibration((640, 480), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1)
        with patch("main.filedialog.askopenfilename", return_value="file.json"), \
             patch("main.load_calibration", return_value=value), \
             patch("main.messagebox.askyesno", return_value=False) as confirm, \
             patch("main.make_maps") as maps:
            self.app.load()
        confirm.assert_called_once()
        maps.assert_not_called()

    def test_extrinsics_requires_intrinsics(self):
        self.app.busy = False
        with patch("main.messagebox.showerror") as error, patch.object(self.app, "run_job") as job:
            self.app.frame = np.zeros((480, 640, 3), np.uint8)
            self.app.start_extrinsics()
        error.assert_called_once()
        self.assertIn("intrinsics", error.call_args.args[1])
        job.assert_not_called()

    def test_ground_params_validation(self):
        self.app.square_mm = Mock(get=Mock(return_value="-5"))
        self.app.origin_x = Mock(get=Mock(return_value="0"))
        self.app.origin_y = Mock(get=Mock(return_value="1500"))
        self.app.board_angle = Mock(get=Mock(return_value="0"))
        with self.assertRaises(ValueError):
            self.app.ground_params()
        self.app.square_mm.get.return_value = "100"
        self.assertEqual(self.app.ground_params(), (100.0, (0.0, 1500.0), 0.0))

    def test_extrinsics_event_applies_result(self):
        self.app.calibration = Calibration((640, 480), (9, 6), np.eye(3), np.zeros((4, 1)), 0.1)
        self.app.busy = True
        value = Mock(pattern=(9, 6), square_mm=25.4, origin_mm=(0.0, 1000.0), angle_deg=0.0)
        self.app.view_mm = Mock(get=Mock(return_value="2"))
        with patch("main.make_ground_maps", return_value=("mx", "my")) as maps:
            value.rms = 0.5
            value.position_mm.return_value = (0, -1000, 1200)
            self.app.handle_event(("extrinsics", 4, value, None))
        self.assertIs(self.app.extrinsics, value)
        scale, center = maps.call_args.args[2], maps.call_args.args[4]
        self.assertAlmostEqual(scale, 0.5)
        self.assertAlmostEqual(center[0], 25.4 * 4)
        self.assertAlmostEqual(center[1], 1000 - 25.4 * 2.5)
        self.assertEqual(self.app.ground_maps, ("mx", "my"))
        self.assertFalse(self.app.busy)


class WorkerTests(unittest.TestCase):
    def test_worker_recovers_after_open_exception(self):
        events = queue.Queue()
        worker = CameraWorker(events)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
        with patch("main.cv2.VideoCapture", side_effect=[RuntimeError("open failed"), capture]):
            worker.start()
            try:
                worker.commands.put(("open", 0, 1))
                error = events.get(timeout=3)
                self.assertIn("open failed", error[2])
                worker.commands.put(("open", 1, 2))
                token, frame = worker.frames.get(timeout=3)
                self.assertEqual(token, 2)
                self.assertEqual(frame.shape, (4, 4, 3))
            finally:
                worker.commands.put(("stop", None, 2))
                worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            capture.release.assert_called()


if __name__ == "__main__":
    unittest.main()
