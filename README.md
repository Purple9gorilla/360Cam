# 360Cam

A homemade bird's-eye (surround-view) camera system for a car. Several cameras are calibrated, warped onto the ground plane and stitched into one top-down view.

## Setup

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

OpenCV must stay below version 5. OpenCV 5.0's fisheye calibration is broken. Run everything with `.\.venv\Scripts\python.exe` from the project root.

## Workflow

### 1. Check your cameras
```
python Calibration\MultiCapture.py
python Calibration\Capture.py
```
`MultiCapture.py` shows every connected camera in one grid. Note which index is which position.

### 2. Intrinsics (once per camera and lens)
Corrects lens distortion. Hold a printed checkerboard in front of the camera and tilt and move it around, covering the edges and corners. Frames save automatically.

```
python Calibration\CalibrateGUI.py
```
- Set the options in the left panel, then click **Find cameras**. Every connected camera appears in one grid. Click a camera (or press its number key) to calibrate it. When it finishes (or you press Backspace), you return to the grid, and calibrated cameras are marked.
- **Names** (`0=front 1=rear`) sets the output file names. Unnamed cameras are called `cam<index>`.
- Tick **Pinhole** for lenses up to about 120°. Leave it off for real fisheye lenses (about 160–190°).
- **Inner corners** is the number of inner checkerboard corners (default 9 × 6). **Square size** is in mm.
- Auto-saves when the board has moved at least *Min corner movement* px and *Min seconds between saves* have passed, then calibrates after *Frames* frames (default 20). SPACE forces a save, `c` calibrates now, Backspace goes back.
- Sanity check: the printed RMS should be under about 1px, and the focal length should be near `(width / 2) / tan(HFOV / 2)` (about 540px for a 100° lens at 1280px wide).
- Output: `calib\<name>.npz`.

Check the result:
```
python Calibration\PreviewUndistort.py --camera 0 --name front
```
Straight real-world lines should look straight on the right-hand side.

Redo this whenever you change the lens or the resolution.

### 3. Ground plane (once per camera, after it is mounted)
Maps each camera's view onto the floor. Lay a large checkerboard flat on the floor at a known position.

```
python Calibration\CalibrateGround.py --camera 0 --name front --origin -400 1500 --square 100
```
- Coordinates are in mm from the car centre: x = right, y = forward.
- `--origin X Y` is where the board's first corner sits. `--angle` rotates the board (degrees, counter-clockwise from above).
- Press SPACE when the board is detected. A live top-down preview opens. The board should look square and sit where you placed it.
- Use the same `--scale` and `--canvas-mm` for every camera.
- Output: `calib\<name>_ground.npz`.

Redo this if a camera is moved or re-aimed.

### 4. Bird's-eye view
```
python BirdsEye.py --cam front=0 rear=1 left=2 right=3
```
- Press `q` to quit. `--snapshot out.png` saves one image and exits.
- `--car-mm W L` sets the car outline size. `--no-gain` turns off brightness matching.
- Overlaps are blended with feathered weights. Brightness between cameras is matched automatically where their views overlap.

### Try it without hardware
```
python BirdsEye.py --simulate
```
Renders four simulated cameras around a virtual car looking at a tile floor.

## Project layout

| File | Purpose |
|---|---|
| `Calibration\Capture.py` | Single camera live view |
| `Calibration\MultiCapture.py` | All cameras in one grid |
| `Calibration\CalibrateGUI.py` | Lens calibration GUI, saves frames automatically |
| `Calibration\PreviewUndistort.py` | Raw vs undistorted preview |
| `Calibration\CalibrateGround.py` | Ground-plane homography per camera |
| `BirdsEye.py` | Stitches the cameras into the live top-down view |
| `Simulation.py` | Synthetic cameras for testing |
| `calib\` | Saved calibration files |

## Known limits

- Narrow lenses (about 100°) leave gaps at the corners of the car. Wider lenses, or mounting the cameras higher, fill them.
- Only flat ground is modelled. Tall objects appear stretched and smeared away from the car.
