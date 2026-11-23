"""Fisheye checkerboard calibration and validated JSON persistence."""

from dataclasses import dataclass
import json
import re
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
    samples = list(samples)
    while True:
        try:
            rms, matrix, distortion, _, _ = cv2.fisheye.calibrate(
                [obj.copy() for _ in samples], samples, size, None, None,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            break
        except cv2.error as exc:
            # OpenCV names the ill-conditioned view; drop it and retry.
            match = re.search(r"input array (\d+)", str(exc))
            if not match or int(match.group(1)) >= len(samples):
                raise
            if len(samples) - 1 < MIN_SAMPLES:
                raise ValueError(
                    f"Some views are degenerate and fewer than {MIN_SAMPLES} remain. "
                    "Reset and capture more varied tilts and positions."
                ) from exc
            del samples[int(match.group(1))]
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


def make_maps(value, balance=0.5):
    validate(value)
    new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        value.matrix, value.distortion, value.size, np.eye(3), balance=balance
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


GROUND_SCALE = 0.1  # preview pixels per mm
GROUND_CANVAS = (700, 700)  # preview size in pixels, car centre at the middle


def rotated_180(calibration):
    """Intrinsics for the same camera with its image turned upside down."""
    matrix = calibration.matrix.copy()
    matrix[0, 2] = calibration.size[0] - 1 - matrix[0, 2]
    matrix[1, 2] = calibration.size[1] - 1 - matrix[1, 2]
    return Calibration(calibration.size, calibration.pattern, matrix, calibration.distortion,
                       calibration.rms, calibration.device_path, calibration.instance_id)


@dataclass
class Extrinsics:
    size: tuple[int, int]
    pattern: tuple[int, int]
    square_mm: float
    origin_mm: tuple[float, float]
    angle_deg: float
    rvec: np.ndarray
    tvec: np.ndarray
    rms: float
    device_path: str = ""
    instance_id: str = ""
    rotated: bool = False  # camera is mounted upside down; the pose is for the 180-degree turned image

    def rotation(self):
        return cv2.Rodrigues(self.rvec)[0]

    def position_mm(self):
        """Camera position in car coordinates: x right, y forward, z up."""
        return (-self.rotation().T @ self.tvec).ravel()


def board_points(pattern, square_mm, origin_mm, angle_deg):
    """Checkerboard corners in car coordinates (mm, z = 0)."""
    # Columns run right and rows run toward the car/back, so the view from above is not mirrored.
    grid = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm * [1.0, -1.0]
    angle = np.radians(angle_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = grid @ rot.T + np.asarray(origin_mm, np.float64)
    return np.column_stack([xy, np.zeros(len(xy))]).reshape(-1, 1, 3)


def solve_extrinsics(corners, calibration, square_mm, origin_mm, angle_deg,
                     device_path="", instance_id="", rotated=False):
    """Solve the camera pose. For an upside-down camera pass rotated=True and corners
    detected in the image after turning it 180 degrees."""
    validate(calibration)
    pattern = calibration.pattern
    if rotated:
        calibration = rotated_180(calibration)
    if corners.shape != (pattern[0] * pattern[1], 1, 2) or not np.isfinite(corners).all():
        raise ValueError("Detected checkerboard corners are invalid.")
    if not np.isfinite(square_mm) or square_mm <= 0 or not np.isfinite([*origin_mm, angle_deg]).all():
        raise ValueError("Square size must be positive and the origin and angle must be numbers.")
    obj = board_points(pattern, square_mm, origin_mm, angle_deg)
    undistorted = cv2.fisheye.undistortPoints(
        corners.astype(np.float64), calibration.matrix, calibration.distortion, P=calibration.matrix
    )
    ok, rvec, tvec = cv2.solvePnP(obj, undistorted, calibration.matrix, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("Could not solve the camera pose from this view.")
    rvec, tvec = cv2.solvePnPRefineLM(obj, undistorted, calibration.matrix, None, rvec, tvec)
    projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, calibration.matrix, calibration.distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=-1))))
    result = Extrinsics(
        calibration.size, pattern, float(square_mm), (float(origin_mm[0]), float(origin_mm[1])),
        float(angle_deg), rvec.reshape(3, 1), tvec.reshape(3, 1), rms, device_path, instance_id, bool(rotated),
    )
    validate_extrinsics(result)
    if result.position_mm()[2] <= 0:
        raise ValueError(
            "The solved camera is below the floor, so the board orientation is wrong. Check which corner is "
            "the origin (top-left as the camera sees it), the board angle, and the origin X/Y."
        )
    return result


def validate_extrinsics(value):
    if (len(value.size) != 2 or any(type(x) is not int or x <= 0 for x in value.size)
            or len(value.pattern) != 2 or any(type(x) is not int or not 2 <= x <= 100 for x in value.pattern)):
        raise ValueError("Invalid image size or checkerboard dimensions.")
    if (value.rvec.shape != (3, 1) or value.tvec.shape != (3, 1)
            or not np.isfinite(value.rvec).all() or not np.isfinite(value.tvec).all()
            or not np.isfinite(value.rms) or value.rms < 0
            or not np.isfinite(value.square_mm) or value.square_mm <= 0
            or len(value.origin_mm) != 2 or not np.isfinite([*value.origin_mm, value.angle_deg]).all()):
        raise ValueError("Invalid extrinsic parameters.")
    if not isinstance(value.device_path, str) or not isinstance(value.instance_id, str):
        raise ValueError("Invalid camera identity.")
    if not isinstance(value.rotated, bool):
        raise ValueError("Invalid camera orientation.")


def make_ground_maps(calibration, extrinsics, scale=GROUND_SCALE, canvas=GROUND_CANVAS, center_mm=(0.0, 0.0)):
    """Remap tables turning a raw camera frame into a top-down view of the floor."""
    validate(calibration)
    validate_extrinsics(extrinsics)
    pose_calibration = rotated_180(calibration) if extrinsics.rotated else calibration
    width, height = canvas
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    ground = np.stack([center_mm[0] + (u - width / 2) / scale,
                       center_mm[1] + (height / 2 - v) / scale, np.zeros_like(u)], axis=-1)
    ground = ground.reshape(-1, 1, 3)
    in_front = (ground[:, 0] @ extrinsics.rotation().T + extrinsics.tvec.T)[:, 2] > 1e-6
    projected, _ = cv2.fisheye.projectPoints(
        ground, extrinsics.rvec, extrinsics.tvec, pose_calibration.matrix, pose_calibration.distortion
    )
    projected = projected.reshape(height, width, 2)
    if extrinsics.rotated:
        projected[..., 0] = calibration.size[0] - 1 - projected[..., 0]
        projected[..., 1] = calibration.size[1] - 1 - projected[..., 1]
    projected[~in_front.reshape(height, width)] = -1
    return projected[..., 0].astype(np.float32), projected[..., 1].astype(np.float32)


def save_extrinsics(path, value):
    validate_extrinsics(value)
    data = {
        "version": 1, "image_size": value.size, "pattern": value.pattern,
        "square_mm": value.square_mm, "origin_mm": value.origin_mm, "angle_deg": value.angle_deg,
        "rvec": value.rvec.ravel().tolist(), "tvec": value.tvec.ravel().tolist(),
        "rms": value.rms, "device_path": value.device_path, "instance_id": value.instance_id,
        "rotated": value.rotated,
    }
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_extrinsics(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data["version"] != 1:
            raise ValueError("Unsupported extrinsics file version.")
        result = Extrinsics(
            tuple(data["image_size"]), tuple(data["pattern"]), float(data["square_mm"]),
            tuple(float(x) for x in data["origin_mm"]), float(data["angle_deg"]),
            np.asarray(data["rvec"], dtype=np.float64).reshape(3, 1),
            np.asarray(data["tvec"], dtype=np.float64).reshape(3, 1), float(data["rms"]),
            data.get("device_path", ""), data.get("instance_id", ""), data.get("rotated", False),
        )
        validate_extrinsics(result)
        return result
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid extrinsics file: {exc}") from exc
