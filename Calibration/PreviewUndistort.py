import argparse
import os
import sys

import cv2
import numpy as np

WINDOW = 'Undistort Preview (left: raw, right: undistorted)'
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calib')


def build_maps(K, D, size, fisheye, balance):
    if fisheye:
        D = D.reshape(-1)[:4].reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=balance)
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, balance)
    return cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)


def main():
    parser = argparse.ArgumentParser(description="Preview a camera's undistorted view from a calibration file.")
    parser.add_argument('--camera', type=int, default=0, help="camera index")
    parser.add_argument('--name', help="calibration name (default: cam<index>)")
    parser.add_argument('--dir', default=DEFAULT_DIR, help="calibration directory")
    parser.add_argument('--balance', type=float, default=0.5,
                        help="0 = crop to valid pixels, 1 = keep the whole field of view")
    args = parser.parse_args()

    path = os.path.join(args.dir, f'{args.name or f"cam{args.camera}"}.npz')
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run CalibrateIntrinsics.py first.")
        sys.exit(1)
    data = np.load(path)
    K, D, size, fisheye = data['K'], data['D'], tuple(int(v) for v in data['size']), bool(data['fisheye'])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        sys.exit(1)

    map1, map2 = build_maps(K, D, size, fisheye, args.balance)
    print("Straight real-world lines should look straight on the right. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        if (frame.shape[1], frame.shape[0]) != size:
            print(f"Error: camera gives {frame.shape[1]}x{frame.shape[0]}, calibration is {size[0]}x{size[1]}.")
            break
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow(WINDOW, np.hstack([cv2.resize(frame, (640, 360)), cv2.resize(undistorted, (640, 360))]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
