# Fisheye camera calibration

A Windows application for calibrating a camera with a printed checkerboard using OpenCV's fisheye model. It shows the original and corrected live camera images side by side.

## Setup

Install Python with Tkinter, then in this folder run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Calibrate

1. Connect the camera, click **Refresh** if needed, and select it from the dropdown. The app uses the Windows DirectShow backend.
2. Print and mount a flat checkerboard. Enter the number of **inner corners**, not squares, across and down. For example, 10 by 7 squares have 9 by 6 inner corners.
3. Keep the whole board visible and click **Capture view** for each image. Capture at least 10 different views; 15 or more are recommended. Move and tilt the board across the image, including near its edges. Avoid blurry or repeated views.
4. Click **Calibrate**. Check the corrected preview and reported RMS reprojection error. If calibration fails or the result looks wrong, reset and capture more varied, clear views.
5. Click **Save** to store a JSON file. To reuse it, select the camera and click **Load**. The resolution must match. If the saved device path differs from the selected camera, or either path is missing, the app asks for confirmation before loading.

## Extrinsics (camera pose on the car)

Once the intrinsics are calibrated or loaded, the extrinsics give the camera's position and angle relative to the car, so views can later be stitched into a top-down image.

1. Mount the camera. Lay a large checkerboard flat on the floor, using the same inner-corner counts as the intrinsics.
2. Measure where the board's first corner (top-left, as the upright view shows it) sits, in mm from the car centre: **X** is right and **Y** is forward. The board extends right and *back* (down in the image) from that corner. Enter the printed **square** size in mm, the **origin X/Y**, and the **angle** (degrees, counter-clockwise from above).
3. Keep the whole board visible and click **Solve extrinsics**. The result shows the reprojection RMS and the camera position and height. The **Ground view** pane shows a top-down view of the floor (700 px wide at 0.1 px/mm); the board should look square and sit where you placed it.
4. Click **Save extrinsics** to store a JSON file, or **Load extrinsics** to reuse one. Loading needs the matching intrinsics loaded first.

Redo this if the camera is moved or re-aimed. Only flat ground is modelled, so tall objects look stretched. The **view mm/px** field zooms the Ground view, which is centred on the board.

**Upside-down cameras:** tick **Camera upside down** before solving. The app turns the frame 180° for detection, so measure the board's first corner as the upright view shows it. The stitcher handles the rotation automatically, and intrinsics don't need redoing. Solving fails if the result puts the camera below the floor, which usually means the corner, angle or this checkbox is wrong.

## Stitch a top-down view

```powershell
.\.venv\Scripts\python.exe stitch.py
```

1. Click **Add camera** for each camera. In each row, pick the camera, then load its **Intrinsics** and **Extrinsics** JSON files. The extrinsics must have been solved with those intrinsics, and every row needs a different camera.
2. Set **mm per px** (default 10), **canvas px** (square, 100 to 2000) and the **car W x L** in mm for the outline.
3. Click **Start**. Camera resolutions must match their calibrations. Overlaps are feather-blended, and **Match brightness** evens out exposure differences where views overlap. Areas no camera sees stay black.
4. **Snapshot** saves the current view as a PNG. **Stop** before removing a camera.

Only flat ground is modelled, so tall objects smear away from the car.

Changing the camera, checkerboard dimensions, or camera resolution clears the current samples and calibration. Reset clears them as well. Undistortion is unavailable until a calibration is applied. The program does not save captured images.

The dropdown shows DirectShow names and Windows hardware IDs when available. Hardware IDs usually identify a model, not an individual device; the separate device instance and interface path help distinguish devices. OpenCV accesses cameras by DirectShow index. The app assumes its ordering matches Windows enumeration but cannot verify that mapping, and ordering may change when devices reconnect. Refresh and visually verify the selected camera before calibrating or loading a file; device-path checks cannot prove the identity of the camera OpenCV actually opened. Some virtual devices do not expose Windows identifiers.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
