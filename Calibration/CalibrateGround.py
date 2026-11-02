import argparse
import os
import sys

import cv2
import numpy as np

WINDOW = 'Ground Calibration'
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calib')


def ground_to_canvas(pts_mm, scale, canvas):
    """Ground frame: origin at car centre, x right, y forward, in mm."""
    out = np.empty_like(pts_mm, dtype=np.float64)
    out[:, 0] = pts_mm[:, 0] * scale + canvas[0] / 2
    out[:, 1] = canvas[1] / 2 - pts_mm[:, 1] * scale
    return out


def board_ground_points(board, square, origin, angle_deg):
    """Ground positions (mm) of the board's inner corners, in OpenCV corner order."""
    local = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2).astype(np.float64) * square
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return local @ rot.T + np.asarray(origin, dtype=np.float64)


def to_normalized(pixels, K, D, fisheye):
    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    if fisheye:
        return cv2.fisheye.undistortPoints(pts, K, D.reshape(-1)[:4].reshape(4, 1)).reshape(-1, 2)
    return cv2.undistortPoints(pts, K, D).reshape(-1, 2)


def compute_homography(pixels, ground_mm, K, D, fisheye, scale, canvas):
    """Homography from normalized camera coordinates to top-down canvas pixels."""
    norm = to_normalized(pixels, K, D, fisheye)
    H, mask = cv2.findHomography(norm, ground_to_canvas(ground_mm, scale, canvas), cv2.RANSAC, 2.0)
    return H


def build_ground_maps(K, D, fisheye, H, canvas):
    """Single remap from the raw camera image straight to the top-down canvas."""
    xs, ys = np.meshgrid(np.arange(canvas[0]), np.arange(canvas[1]))
    grid = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0).astype(np.float64)
    norm = np.linalg.inv(H) @ grid
    norm = (norm[:2] / norm[2]).T
    obj = np.hstack([norm, np.ones((len(norm), 1))]).reshape(-1, 1, 3)
    zero = np.zeros(3)
    if fisheye:
        pix, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, D.reshape(-1)[:4].reshape(4, 1))
    else:
        pix, _ = cv2.projectPoints(obj, zero, zero, K, D)
    pix = pix.reshape(canvas[1], canvas[0], 2).astype(np.float32)
    # Points behind the camera must not wrap into the image
    pix[(obj[:, 0, 2] <= 0).reshape(canvas[1], canvas[0])] = -1
    return pix[..., 0], pix[..., 1]


def find_corners(gray, board):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera's top-down (ground plane) view using a checkerboard lying flat on the floor.")
    parser.add_argument('--camera', type=int, default=0, help="camera index")
    parser.add_argument('--name', help="calibration name (default: cam<index>)")
    parser.add_argument('--dir', default=DEFAULT_DIR, help="calibration directory")
    parser.add_argument('--board', type=int, nargs=2, default=(9, 6), metavar=('COLS', 'ROWS'),
                        help="inner corners of the checkerboard (default: 9 6)")
    parser.add_argument('--square', type=float, default=25.0, help="square size in mm")
    parser.add_argument('--origin', type=float, nargs=2, required=True, metavar=('X', 'Y'),
                        help="ground position in mm of the board's first corner, relative to the car centre "
                             "(x = right, y = forward)")
    parser.add_argument('--angle', type=float, default=0.0,
                        help="rotation in degrees (counter-clockwise from above) of the board's column direction "
                             "relative to the car's right (+x) axis")
    parser.add_argument('--scale', type=float, default=0.2, help="canvas pixels per mm")
    parser.add_argument('--canvas-mm', type=float, nargs=2, default=(6000, 6000), metavar=('W', 'H'),
                        help="ground area shown, in mm (default: 6000 6000)")
    args = parser.parse_args()
    name = args.name or f'cam{args.camera}'

    path = os.path.join(args.dir, f'{name}.npz')
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run CalibrateGUI.py first.")
        sys.exit(1)
    data = np.load(path)
    K, D, size, fisheye = data['K'], data['D'], tuple(int(v) for v in data['size']), bool(data['fisheye'])
    canvas = (int(args.canvas_mm[0] * args.scale), int(args.canvas_mm[1] * args.scale))
    board = tuple(args.board)
    ground = board_ground_points(board, args.square, args.origin, args.angle)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        sys.exit(1)

    print("Lay the board flat on the floor at the given --origin/--angle.")
    print("SPACE: compute ground calibration  q: quit")
    H, maps = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        if (frame.shape[1], frame.shape[0]) != size:
            print(f"Error: camera gives {frame.shape[1]}x{frame.shape[0]}, calibration is {size[0]}x{size[1]}.")
            break
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board)
        view = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(view, board, corners, True)
        cv2.putText(view, f'{name}: ' + ('board found - SPACE to calibrate' if corners is not None else 'no board'),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)

        if H is not None:
            top = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            cv2.drawMarker(top, (canvas[0] // 2, canvas[1] // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.imshow('Top-down preview', top)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and corners is not None:
            H = compute_homography(corners, ground, K, D, fisheye, args.scale, canvas)
            if H is None:
                print("Failed to compute the homography.")
                continue
            maps = build_ground_maps(K, D, fisheye, H, canvas)
            out = os.path.join(args.dir, f'{name}_ground.npz')
            os.makedirs(args.dir, exist_ok=True)
            np.savez(out, H=H, scale=args.scale, canvas=np.array(canvas))
            print(f"Saved {out}. The board should look square and correctly placed in the preview. "
                  f"Press SPACE to recompute, q to quit.")
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
