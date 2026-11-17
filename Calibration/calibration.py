"""Fisheye checkerboard calibration and validated JSON persistence."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

MIN_SAMPLES = 10


@dataclass
class Calibration:
    size: tuple[int, int]
    pattern: tuple[int, int]
    matrix: np.ndarray
    distortion: np.ndarray
    rms: float
    device_path: str = ""
    instance_id: str = ""


def find_corners(frame, pattern):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    return cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    ).astype(np.float64).reshape(-1, 1, 2)


def calibrate(samples, pattern, size, device_path="", instance_id=""):
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"Capture at least {MIN_SAMPLES} valid checkerboard views.")
    if len(pattern) != 2 or min(pattern) < 2:
        raise ValueError("Checkerboard dimensions must both be at least 2.")
    count = pattern[0] * pattern[1]
    if any(sample.shape != (count, 1, 2) or not np.isfinite(sample).all()
           for sample in samples):
        raise ValueError("Captured checkerboard corners are invalid.")
    obj = np.zeros((count, 1, 3), np.float64)
    obj[:, 0, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    rms, matrix, distortion, _, _ = cv2.fisheye.calibrate(
        [obj.copy() for _ in samples], samples, size, None, None,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    result = Calibration(size, pattern, matrix, distortion, float(rms), device_path, instance_id)
    validate(result)
    return result


def validate(value):
    if (len(value.size) != 2 or any(type(x) is not int or x <= 0 for x in value.size)
            or len(value.pattern) != 2 or any(type(x) is not int or not 2 <= x <= 100 for x in value.pattern)):
        raise ValueError("Invalid image size or checkerboard dimensions.")
    if (value.matrix.shape != (3, 3) or value.distortion.shape != (4, 1)
            or not np.isfinite(value.matrix).all() or not np.isfinite(value.distortion).all()
            or not np.isfinite(value.rms) or value.rms < 0
            or value.matrix[0, 0] <= 0 or value.matrix[1, 1] <= 0
            or not np.allclose(value.matrix[2], [0, 0, 1])):
        raise ValueError("Invalid fisheye calibration parameters.")
    if not isinstance(value.device_path, str) or not isinstance(value.instance_id, str):
        raise ValueError("Invalid camera identity.")


def make_maps(value):
    validate(value)
    new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        value.matrix, value.distortion, value.size, np.eye(3), balance=0.0
    )
    return cv2.fisheye.initUndistortRectifyMap(
        value.matrix, value.distortion, np.eye(3), new_matrix, value.size, cv2.CV_16SC2
    )


def save_calibration(path, value):
    validate(value)
    data = {
        "version": 1, "image_size": value.size, "pattern": value.pattern,
        "camera_matrix": value.matrix.tolist(), "distortion": value.distortion.tolist(),
        "rms": value.rms, "device_path": value.device_path, "instance_id": value.instance_id,
    }
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_calibration(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data["version"] != 1:
            raise ValueError("Unsupported calibration file version.")
        result = Calibration(
            tuple(data["image_size"]), tuple(data["pattern"]),
            np.asarray(data["camera_matrix"], dtype=np.float64),
            np.asarray(data["distortion"], dtype=np.float64), float(data["rms"]),
            data.get("device_path", ""), data.get("instance_id", ""),
        )
        validate(result)
        return result
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid calibration file: {exc}") from exc
