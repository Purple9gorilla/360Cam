import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from Calibration import CalibrateGround as G

WINDOW = "Bird's-eye view"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib')
FEATHER_PX = 60


class StaticSource:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame

    def release(self):
        pass


class ThreadedCapture:
    """Keeps only the newest frame so a slow camera never delays the others."""

    def __init__(self, index, size):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.frame, self.running = None, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


class Stitcher:
    MIN_OVERLAP_PX = 500
    GAIN_EVERY = 10
    GAIN_SMOOTH = 0.2
    GAIN_REG = 0.05

    def __init__(self, cams, canvas, gain=True):
        self.canvas = canvas
        self.maps, self.weights = [], []
        for cam in cams:
            mx, my = G.build_ground_maps(cam['K'], cam['D'], cam['fisheye'], cam['H'], canvas)
            h, w = cam['size'][1], cam['size'][0]
            valid = ((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)).astype(np.uint8)
            dist = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
            self.weights.append(np.minimum(dist, FEATHER_PX) / FEATHER_PX)
            self.maps.append(cv2.convertMaps(mx, my, cv2.CV_16SC2))
        n = len(cams)
        self.pairs = []
        if gain:
            for i in range(n):
                for j in range(i + 1, n):
                    mask = (self.weights[i] > 0.5) & (self.weights[j] > 0.5)
                    if mask.sum() >= self.MIN_OVERLAP_PX:
                        self.pairs.append((i, j, mask))
        self.gains = np.ones((n, 3), np.float32)
        self.frame_count = 0

    def update_gains(self, warped):
        """Least-squares per-camera, per-channel gains that make overlapping views agree."""
        n = len(warped)
        rows, rhs = [], []
        eqs = [[] for _ in range(3)]
        for i, j, mask in self.pairs:
            if warped[i] is None or warped[j] is None:
                continue
            mi = warped[i][mask].mean(axis=0) + 1e-3
            mj = warped[j][mask].mean(axis=0) + 1e-3
            scale = (mi + mj) / 2
            for c in range(3):
                row = np.zeros(n)
                row[i], row[j] = mi[c] / scale[c], -mj[c] / scale[c]
                eqs[c].append(row)
        if not eqs[0]:
            return
        reg = np.sqrt(self.GAIN_REG) * np.eye(n)
        target = np.ones(n) * np.sqrt(self.GAIN_REG)
        for c in range(3):
            A = np.vstack(eqs[c] + [reg])
            b = np.concatenate([np.zeros(len(eqs[c])), target])
            g = np.linalg.lstsq(A, b, rcond=None)[0]
            self.gains[:, c] += self.GAIN_SMOOTH * (np.clip(g, 0.5, 2.0) - self.gains[:, c])

    def stitch(self, frames):
        warped = [None if f is None else cv2.remap(f, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                  for f, (m1, m2) in zip(frames, self.maps)]
        if self.pairs and self.frame_count % self.GAIN_EVERY == 0:
            self.update_gains(warped)
        self.frame_count += 1
        acc = np.zeros((self.canvas[1], self.canvas[0], 3), np.float32)
        total = np.zeros((self.canvas[1], self.canvas[0]), np.float32)
        for img, weight, gain in zip(warped, self.weights, self.gains):
            if img is None:
                continue
            acc += img * (weight[..., None] * gain)
            total += weight
        out = acc / np.maximum(total, 1e-6)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


def draw_car(image, scale, car_mm):
    h, w = image.shape[:2]
    half = (int(car_mm[0] * scale / 2), int(car_mm[1] * scale / 2))
    tl, br = (w // 2 - half[0], h // 2 - half[1]), (w // 2 + half[0], h // 2 + half[1])
    cv2.rectangle(image, tl, br, (50, 50, 50), -1)
    cv2.rectangle(image, tl, br, (255, 255, 255), 2)
    cv2.arrowedLine(image, (w // 2, h // 2 + half[1] // 2), (w // 2, h // 2 - half[1] // 2),
                    (255, 255, 255), 2, tipLength=0.3)


def load_camera(name, index, directory):
    paths = [os.path.join(directory, f'{name}.npz'), os.path.join(directory, f'{name}_ground.npz')]
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"Error: {p} not found. Run CalibrateIntrinsics.py and CalibrateGround.py for '{name}'.")
    intr, ground = np.load(paths[0]), np.load(paths[1])
    size = tuple(int(v) for v in intr['size'])
    cam = {'name': name, 'K': intr['K'], 'D': intr['D'], 'fisheye': bool(intr['fisheye']),
           'H': ground['H'], 'size': size, 'scale': float(ground['scale']),
           'canvas': tuple(int(v) for v in ground['canvas'])}
    cam['source'] = ThreadedCapture(index, size)
    return cam


def main():
    parser = argparse.ArgumentParser(description="Live bird's-eye view from calibrated cameras.")
    parser.add_argument('--cam', nargs='+', metavar='NAME=INDEX',
                        help="cameras to use, e.g. front=0 rear=1 left=2 right=3")
    parser.add_argument('--dir', default=DEFAULT_DIR, help="calibration directory")
    parser.add_argument('--car-mm', type=float, nargs=2, default=(1800, 4500), metavar=('W', 'L'),
                        help="car width and length in mm for the overlay")
    parser.add_argument('--simulate', action='store_true', help="use synthetic cameras instead of hardware")
    parser.add_argument('--no-gain', action='store_true', help="disable brightness matching between cameras")
    parser.add_argument('--snapshot', metavar='PATH', help="save one stitched image and exit")
    args = parser.parse_args()

    if args.simulate:
        import Simulation
        scale, canvas = 0.2, (1200, 1200)
        cams = Simulation.make_cameras(canvas, scale, car_mm=args.car_mm)
        for cam in cams:
            cam['size'] = (cam['frame'].shape[1], cam['frame'].shape[0])
            cam['source'] = StaticSource(cam['frame'])
    elif args.cam:
        cams = []
        for item in args.cam:
            name, _, index = item.partition('=')
            if not index.isdigit():
                sys.exit(f"Error: '{item}' should look like front=0")
            cams.append(load_camera(name, int(index), args.dir))
        scale, canvas = cams[0]['scale'], cams[0]['canvas']
        if any(c['canvas'] != canvas or c['scale'] != scale for c in cams):
            sys.exit("Error: all cameras must be ground-calibrated with the same --scale and --canvas-mm.")
    else:
        parser.error("give --cam NAME=INDEX ... or --simulate")

    stitcher = Stitcher(cams, canvas, gain=not args.no_gain)
    print("Press 'q' to quit.")

    while True:
        frames = [c['source'].read() for c in cams]
        if args.snapshot and any(f is None for f in frames):
            time.sleep(0.05)
            continue
        view = stitcher.stitch(frames)
        draw_car(view, scale, args.car_mm)
        if args.snapshot:
            cv2.imwrite(args.snapshot, view)
            print(f"Saved {args.snapshot}")
            break
        cv2.imshow(WINDOW, view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    for c in cams:
        c['source'].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
