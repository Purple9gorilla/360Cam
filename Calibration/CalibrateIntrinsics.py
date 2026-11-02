import argparse
import math
import os

os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # quiet DSHOW warnings when probing missing camera indices
import sys
import time

import cv2
import numpy as np

WINDOW = 'Intrinsic Calibration'
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calib')
MIN_FRAMES = 10
MAX_PROBE = 10
PREVIEW_RESOLUTIONS = [(640, 480), (320, 240)]
TILE_W, TILE_H = 480, 270


def find_corners(gray, board):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)


def calibrate(obj_points, img_points, size, fisheye):
    if fisheye:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        # OpenCV 5 moved these flags from cv2.fisheye to cv2
        ns = cv2.fisheye if hasattr(cv2.fisheye, 'CALIB_FIX_SKEW') else cv2
        flags = (ns.CALIB_RECOMPUTE_EXTRINSIC | ns.CALIB_CHECK_COND | ns.CALIB_FIX_SKEW)
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3) for o in obj_points], img_points, size, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return rms, K, D


def calibrate_camera(index, name, args):
    """Calibrate one camera. Returns 'saved', 'back' (Backspace/Esc or camera error) or 'quit'."""
    board = tuple(args.board)
    template = np.zeros((board[0] * board[1], 3), np.float32)
    template[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * args.square

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Error: Could not open camera {index}.")
        return 'back'

    obj_points, img_points, size = [], [], None
    last_corners, last_save = None, 0.0
    result = 'back'
    print(f"Calibrating '{name}' (camera {index}). Hold the board in view and move it around; frames save "
          f"automatically (target {args.frames}). SPACE: force save  c: calibrate now  "
          f"Backspace: back  q: quit")

    def save_frame(corners):
        obj_points.append(template.copy())
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32 if args.pinhole else np.float64))
        print(f"Saved frame {len(img_points)}")

    def run_calibration():
        if len(img_points) < MIN_FRAMES:
            print(f"Need at least {MIN_FRAMES} frames, have {len(img_points)}.")
            return False
        try:
            rms, K, D = calibrate(obj_points, img_points, size, not args.pinhole)
        except cv2.error as e:
            print(f"Calibration failed ({e}). Keep capturing more varied frames.")
            return False
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f'{name}.npz')
        np.savez(path, K=K, D=D, size=np.array(size), fisheye=not args.pinhole)
        print(f"RMS error: {rms:.3f}px. Saved to {path}")
        return True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        size = frame.shape[1], frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, board)

        now = time.time()
        auto_saved = False
        if corners is not None and now - last_save >= args.interval and (
                last_corners is None
                or np.linalg.norm(corners - last_corners, axis=2).mean() >= args.min_move):
            save_frame(corners)
            last_corners, last_save, auto_saved = corners, now, True

        view = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(view, board, corners, True)
        if auto_saved:
            cv2.rectangle(view, (0, 0), (size[0] - 1, size[1] - 1), (255, 255, 255), 12)
        status = f'{name}: {len(img_points)}/{args.frames} frames' + ('' if corners is not None else ' (no board)')
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            result = 'quit'
            break
        if key in (8, 27):
            break
        if key == ord(' ') and corners is not None:
            save_frame(corners)
            last_corners, last_save = corners, now
        elif key == ord('c') and run_calibration():
            result = 'saved'
            break
        if len(img_points) >= args.frames and auto_saved and run_calibration():
            result = 'saved'
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            result = 'quit'
            break

    cap.release()
    return result


def find_indices():
    indices = []
    for i in range(MAX_PROBE):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            indices.append(i)
        cap.release()
    return indices


def open_previews(indices):
    # Several USB cameras often can't all stream at high resolution, so step down until all open.
    for width, height in PREVIEW_RESOLUTIONS:
        caps = {}
        for i in indices:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.isOpened() and cap.read()[0]:
                caps[i] = cap
            else:
                cap.release()
        if len(caps) == len(indices):
            return caps
        for cap in caps.values():
            cap.release()
    return {}


def make_tile(index, name, frame, done):
    if frame is None:
        tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    else:
        tile = cv2.resize(frame, (TILE_W, TILE_H))
    color = (0, 200, 0) if done else (0, 255, 255)
    label = f'[{index}] {name}' + ('  CALIBRATED' if done else '')
    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), color, 2)
    return tile


def pick_camera(indices, names, done):
    """Show all camera feeds; return the chosen index or None to quit."""
    caps = open_previews(indices)
    if not caps:
        print("Error: could not open the cameras for preview.")
        return None
    print("Press a camera's number key to calibrate it. q: quit")
    chosen = None
    try:
        while True:
            tiles = []
            for i in indices:
                ret, frame = caps[i].read()
                tiles.append(make_tile(i, names[i], frame if ret else None, i in done))
            cols = math.ceil(math.sqrt(len(tiles)))
            rows = math.ceil(len(tiles) / cols)
            tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
            grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
            cv2.imshow(WINDOW, grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if ord('0') <= key <= ord('9') and key - ord('0') in indices:
                chosen = key - ord('0')
                break
    finally:
        for cap in caps.values():
            cap.release()
    return chosen


def parse_names(items):
    names = {}
    for item in items or []:
        index, _, name = item.partition('=')
        if not (index.isdigit() and name):
            sys.exit(f"Error: '{item}' should look like 0=front")
        names[int(index)] = name
    return names


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics with a checkerboard. "
                                     "Without --camera, shows all cameras and lets you calibrate them one by one.")
    parser.add_argument('--camera', type=int, help="calibrate only this camera index (skips the picker)")
    parser.add_argument('--name', help="name for --camera, e.g. front (default: cam<index>)")
    parser.add_argument('--names', nargs='+', metavar='INDEX=NAME',
                        help="names for the picker, e.g. 0=front 1=rear (default: cam<index>)")
    parser.add_argument('--board', type=int, nargs=2, default=(9, 6), metavar=('COLS', 'ROWS'),
                        help="inner corners of the checkerboard (default: 9 6)")
    parser.add_argument('--square', type=float, default=25.0, help="square size in mm")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--pinhole', action='store_true', help="use the pinhole model instead of fisheye")
    parser.add_argument('--out', default=DEFAULT_DIR, help="output directory")
    parser.add_argument('--frames', type=int, default=20, help="auto-calibrate after this many saved frames")
    parser.add_argument('--interval', type=float, default=1.0, help="min seconds between auto-saves")
    parser.add_argument('--min-move', type=float, default=40.0,
                        help="min average corner movement (px) since the last saved frame")
    args = parser.parse_args()

    if args.camera is not None:
        calibrate_camera(args.camera, args.name or f'cam{args.camera}', args)
        cv2.destroyAllWindows()
        return

    indices = find_indices()
    if not indices:
        print("Error: no cameras found.")
        sys.exit(1)
    names = {i: f'cam{i}' for i in indices}
    names.update({i: n for i, n in parse_names(args.names).items() if i in indices})
    print(f"Found cameras: {indices}")

    done = set()
    while True:
        index = pick_camera(indices, names, done)
        if index is None:
            break
        result = calibrate_camera(index, names[index], args)
        if result == 'saved':
            done.add(index)
        elif result == 'quit':
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
