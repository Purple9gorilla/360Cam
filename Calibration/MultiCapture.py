import argparse
import math
import sys

import cv2
import numpy as np

WINDOW = 'Multi Camera Feed'
TILE_W, TILE_H = 640, 360
MAX_PROBE = 10


RESOLUTIONS = [(1280, 720), (640, 480), (320, 240)]


def find_indices(max_index=MAX_PROBE):
    indices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            indices.append(i)
        cap.release()
    return indices


def open_all(indices, width, height):
    cams = []
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not (cap.isOpened() and cap.read()[0]):
            cap.release()
            for _, c in cams:
                c.release()
            return None
        cams.append((i, cap))
    return cams


def choose_indices(available, requested):
    if requested is None:
        print(f"Available cameras: {available}")
        answer = input("Cameras to view (e.g. 0 2, Enter for all): ").replace(',', ' ').split()
        requested = [int(a) for a in answer if a.isdigit()] or available
    chosen = [i for i in dict.fromkeys(requested) if i in available]
    skipped = [i for i in requested if i not in available]
    if skipped:
        print(f"Ignoring unavailable camera(s): {skipped}")
    return chosen


def detect_cameras(requested=None):
    # Several USB cameras often can't all stream at high resolution
    # (shared bandwidth), so step the resolution down until all open.
    available = find_indices()
    if not available:
        return []
    indices = choose_indices(available, requested)
    for width, height in RESOLUTIONS:
        cams = open_all(indices, width, height)
        if cams:
            print(f"Opened {len(cams)} camera(s) at {width}x{height}.")
            return cams
    return []


def make_tile(index, frame):
    if frame is None:
        tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        label = f'Camera {index}: no signal'
    else:
        tile = cv2.resize(frame, (TILE_W, TILE_H))
        label = f'Camera {index}'
    cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return tile


def make_grid(tiles):
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    tiles = tiles + [blank] * (rows * cols - len(tiles))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])


def main():
    parser = argparse.ArgumentParser(description="View multiple cameras in one window.")
    parser.add_argument('--cameras', type=int, nargs='+', metavar='INDEX',
                        help="camera indices to view (default: prompt)")
    args = parser.parse_args()

    cams = detect_cameras(args.cameras)
    if not cams:
        print("Error: No cameras could be opened.")
        sys.exit(1)

    print(f"Found {len(cams)} camera(s): {[i for i, _ in cams]}. Press 'q' to quit.")

    while True:
        tiles = []
        for index, cap in cams:
            ret, frame = cap.read()
            tiles.append(make_tile(index, frame if ret else None))

        cv2.imshow(WINDOW, make_grid(tiles))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    for _, cap in cams:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
