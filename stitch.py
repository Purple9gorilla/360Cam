"""Live top-down stitcher built from saved intrinsics and extrinsics."""

from dataclasses import dataclass, field
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from calibration import GROUND_CANVAS, GROUND_SCALE, load_calibration, load_extrinsics, make_ground_maps
from camera_devices import discover_cameras

FEATHER_PX = 60  # source-image pixels over which a camera fades out at its borders


def camera_weights(mapx, mapy, size, feather=FEATHER_PX):
    """Blend weight per ground pixel: 0 outside the camera image, rising to 1 away from its border."""
    width, height = size
    border = np.minimum.reduce([mapx, mapy, width - 1 - mapx, height - 1 - mapy])
    return np.clip(border / feather, 0.0, 1.0).astype(np.float32)


def luma(image):
    return image @ np.array([0.114, 0.587, 0.299], np.float32)


def match_gains(warped, weights, min_overlap=200):
    """Per-camera gains that even out brightness where camera views overlap."""
    count = len(warped)
    gains = np.ones(count, np.float32)
    ratios = [[] for _ in range(count)]
    lumas = [luma(image) for image in warped]
    for i in range(count):
        for j in range(i + 1, count):
            overlap = (weights[i] > 0.05) & (weights[j] > 0.05)
            if overlap.sum() < min_overlap:
                continue
            mean_i, mean_j = lumas[i][overlap].mean(), lumas[j][overlap].mean()
            if mean_i > 1 and mean_j > 1:
                ratios[i].append(mean_j / mean_i)
                ratios[j].append(mean_i / mean_j)
    for i, values in enumerate(ratios):
        if values:
            gains[i] = np.exp(0.5 * np.mean(np.log(values)))
    gains = np.clip(gains, 0.5, 2.0)
    return gains / np.exp(np.mean(np.log(gains)))


def compose(warped, weights, gains=None):
    """Feather-blend warped float32 BGR images into one uint8 image; uncovered pixels are black."""
    if not warped:
        raise ValueError("Nothing to compose.")
    total = np.zeros(weights[0].shape, np.float32)
    accumulated = np.zeros(warped[0].shape, np.float32)
    for index, (image, weight) in enumerate(zip(warped, weights)):
        gain = 1.0 if gains is None else float(gains[index])
        accumulated += image * (weight * gain)[..., None]
        total += weight
    result = accumulated / np.maximum(total, 1e-6)[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


@dataclass
class Rig:
    """One camera with its saved calibrations."""
    camera: object = None
    calibration: object = None
    extrinsics: object = None
    intrinsics_path: str = ""
    extrinsics_path: str = ""
    maps: tuple = None
    weights: np.ndarray = None
    warped: np.ndarray = None
    widgets: dict = field(default_factory=dict)

    @property
    def ready(self):
        return self.camera is not None and self.calibration is not None and self.extrinsics is not None


def check_pair(calibration, extrinsics):
    if calibration.size != extrinsics.size or calibration.pattern != extrinsics.pattern:
        raise ValueError("The extrinsics were solved with different intrinsics (resolution or checkerboard).")


class CameraStream(threading.Thread):
    """Keeps only the newest frame from one camera."""

    def __init__(self, index):
        super().__init__(daemon=True)
        self.index = index
        self.lock = threading.Lock()
        self.frame = None
        self.error = ""
        self.stopping = threading.Event()

    def run(self):
        capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        try:
            if not capture.isOpened():
                self.error = "Could not open the camera."
                return
            for _ in range(3):
                capture.set(cv2.CAP_PROP_FOURCC, mjpg)
                if int(capture.get(cv2.CAP_PROP_FOURCC)) == mjpg:
                    break
            else:
                self.error = "Camera would not switch to MJPG."
                return
            while not self.stopping.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.error = "Camera stopped delivering frames."
                    return
                with self.lock:
                    self.frame = frame
        except cv2.error as exc:
            self.error = str(exc)
        finally:
            capture.release()

    def latest(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopping.set()


class StitchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Top-down camera stitcher")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.rigs = []
        self.cameras = []
        self.streams = {}
        self.running = False
        self.closed = False
        self.last_output = None
        self.frame_count = 0
        self.gains = None
        self.scale_mm = tk.StringVar(value=f"{1 / GROUND_SCALE:g}")
        self.canvas_px = tk.StringVar(value=str(GROUND_CANVAS[0]))
        self.car_width = tk.StringVar(value="1800")
        self.car_length = tk.StringVar(value="4500")
        self.match_brightness = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Scanning for cameras...")
        self._build()
        self.add_rig()
        threading.Thread(target=self._discover, daemon=True).start()
        self.timer = self.root.after(30, self.tick)

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Add camera", command=self.add_rig).pack(side="left", padx=2)
        self.start_button = ttk.Button(top, text="Start", command=self.toggle)
        self.start_button.pack(side="left", padx=2)
        ttk.Button(top, text="Snapshot", command=self.snapshot).pack(side="left", padx=2)
        ttk.Label(top, text="mm per px").pack(side="left", padx=(12, 2))
        ttk.Entry(top, textvariable=self.scale_mm, width=5).pack(side="left")
        ttk.Label(top, text="canvas px").pack(side="left", padx=(8, 2))
        ttk.Entry(top, textvariable=self.canvas_px, width=5).pack(side="left")
        ttk.Label(top, text="car W x L mm").pack(side="left", padx=(8, 2))
        ttk.Entry(top, textvariable=self.car_width, width=6).pack(side="left")
        ttk.Entry(top, textvariable=self.car_length, width=6).pack(side="left", padx=2)
        ttk.Checkbutton(top, text="Match brightness", variable=self.match_brightness).pack(side="left", padx=8)
        self.rig_frame = ttk.Frame(self.root, padding=(8, 0))
        self.rig_frame.grid(row=1, column=0, sticky="ew")
        self.view = ttk.Label(self.root, anchor="center", text="Press Start")
        self.view.grid(row=3, column=0, sticky="nsew")
        ttk.Label(self.root, textvariable=self.status, padding=8, wraplength=1100).grid(row=4, column=0, sticky="ew")

    def _discover(self):
        try:
            cameras, warning = discover_cameras()
            message = warning or ("Add the cameras, load their files and press Start." if cameras
                                  else "No cameras found.")
        except Exception as exc:
            cameras, message = [], f"Camera discovery failed: {exc}"
        self.root.after(0, lambda: self._discovered(cameras, message))

    def _discovered(self, cameras, message):
        if self.closed:
            return
        self.cameras = cameras
        for rig in self.rigs:
            rig.widgets["dropdown"]["values"] = [camera.label for camera in cameras]
        self.status.set(message)

    def add_rig(self):
        rig = Rig()
        row = len(self.rigs)
        frame = ttk.Frame(self.rig_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=1)
        variable = tk.StringVar()
        dropdown = ttk.Combobox(frame, textvariable=variable, state="readonly", width=48,
                                values=[camera.label for camera in self.cameras])
        dropdown.pack(side="left")
        dropdown.bind("<<ComboboxSelected>>", lambda _e, r=rig: self.select_camera(r))
        ttk.Button(frame, text="Intrinsics...", command=lambda r=rig: self.load_intrinsics(r)).pack(side="left", padx=2)
        ttk.Button(frame, text="Extrinsics...", command=lambda r=rig: self.load_extrinsics(r)).pack(side="left", padx=2)
        ttk.Button(frame, text="Remove", command=lambda r=rig: self.remove_rig(r)).pack(side="left", padx=2)
        info = tk.StringVar(value="No camera / intrinsics / extrinsics")
        ttk.Label(frame, textvariable=info).pack(side="left", padx=6)
        rig.widgets = {"frame": frame, "dropdown": dropdown, "info": info}
        self.rigs.append(rig)

    def remove_rig(self, rig):
        if self.running:
            messagebox.showerror("Remove", "Stop stitching before removing a camera.")
            return
        rig.widgets["frame"].destroy()
        self.rigs.remove(rig)

    def describe(self, rig):
        parts = [
            "camera" if rig.camera else "no camera",
            "intrinsics" if rig.calibration else "no intrinsics",
            "extrinsics" if rig.extrinsics else "no extrinsics",
        ]
        rig.widgets["info"].set(", ".join(parts))

    def select_camera(self, rig):
        index = rig.widgets["dropdown"].current()
        if index >= 0:
            if self.streams:
                self.stop_all()
            rig.camera = self.cameras[index]
            self.describe(rig)

    def confirm_identity(self, rig, value, what):
        """Warn when a file was saved for another device path; returns True to continue."""
        if rig.camera is None or not value.device_path or not rig.camera.device_path:
            return True
        if value.device_path.casefold() == rig.camera.device_path.casefold():
            return True
        return messagebox.askyesno(
            "Different camera device path",
            f"These {what} were saved for a different camera device path.\nUse them for the selected camera anyway?",
        )

    def load_intrinsics(self, rig):
        path = filedialog.askopenfilename(title="Intrinsics", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            value = load_calibration(path)
            if rig.extrinsics is not None:
                check_pair(value, rig.extrinsics)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Intrinsics", str(exc))
            return
        if self.confirm_identity(rig, value, "intrinsics"):
            rig.calibration, rig.intrinsics_path = value, path
            self.describe(rig)

    def load_extrinsics(self, rig):
        path = filedialog.askopenfilename(title="Extrinsics", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            value = load_extrinsics(path)
            if rig.calibration is not None:
                check_pair(rig.calibration, value)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Extrinsics", str(exc))
            return
        if self.confirm_identity(rig, value, "extrinsics"):
            rig.extrinsics, rig.extrinsics_path = value, path
            self.describe(rig)

    def settings(self):
        try:
            scale_mm = float(self.scale_mm.get())
            canvas = int(self.canvas_px.get())
            car = (float(self.car_width.get()), float(self.car_length.get()))
        except ValueError as exc:
            raise ValueError("Enter numbers for mm per px, canvas px and the car size.") from exc
        if not np.isfinite([scale_mm, *car]).all() or scale_mm <= 0 or min(car) <= 0:
            raise ValueError("Scale and car size must be positive.")
        if not 100 <= canvas <= 2000:
            raise ValueError("Canvas must be between 100 and 2000 px.")
        return 1.0 / scale_mm, (canvas, canvas), car

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        try:
            scale, canvas, _car = self.settings()
            rigs = [rig for rig in self.rigs]
            if not rigs:
                raise ValueError("Add at least one camera.")
            for rig in rigs:
                if not rig.ready:
                    raise ValueError("Every camera needs a camera selection, intrinsics and extrinsics.")
            indices = [rig.camera.index for rig in rigs]
            if len(set(indices)) != len(indices):
                raise ValueError("Each row must use a different camera.")
            for rig in rigs:
                rig.maps = make_ground_maps(rig.calibration, rig.extrinsics, scale, canvas)
                rig.weights = camera_weights(*rig.maps, rig.calibration.size)
        except (ValueError, cv2.error) as exc:
            messagebox.showerror("Start", str(exc))
            return
        self.open_streams([rig.camera.index for rig in rigs])
        self.gains = None
        self.frame_count = 0
        self.running = True
        self.start_button.configure(text="Stop")
        self.status.set("Opening cameras...")

    def open_streams(self, indices):
        for index in indices:
            stream = self.streams.get(index)
            if stream is None or not stream.is_alive():
                self.streams[index] = CameraStream(index)
                self.streams[index].start()

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()
        self.streams = {}
        self.running = False
        self.start_button.configure(text="Start")

    def stop(self):
        self.stop_all()
        self.status.set("Stopped.")

    def stitch_once(self):
        """Compose the latest frames; returns (image, problems)."""
        car = self.settings()[2]
        warped, weights, problems = [], [], []
        for rig in self.rigs:
            stream = self.streams[rig.camera.index]
            if stream.error:
                problems.append(f"{rig.camera.name}: {stream.error}")
                continue
            frame = stream.latest()
            if frame is None:
                continue
            if (frame.shape[1], frame.shape[0]) != rig.calibration.size:
                problems.append(f"{rig.camera.name}: frame is {frame.shape[1]}x{frame.shape[0]}, "
                                f"calibration is {rig.calibration.size[0]}x{rig.calibration.size[1]}")
                continue
            warped.append(cv2.remap(frame, *rig.maps, cv2.INTER_LINEAR).astype(np.float32))
            weights.append(rig.weights)
        if not warped:
            return None, problems
        if self.match_brightness.get() and len(warped) > 1:
            if self.gains is None or self.frame_count % 30 == 0:
                self.gains = match_gains(warped, weights)
            gains = self.gains if len(self.gains) == len(warped) else None
        else:
            gains = None
        image = compose(warped, weights, gains)
        self.draw_car(image, car)
        return image, problems

    def draw_car(self, image, car):
        scale = 1.0 / float(self.scale_mm.get())
        height, width = image.shape[:2]
        half_w, half_l = car[0] * scale / 2, car[1] * scale / 2
        top_left = (int(width / 2 - half_w), int(height / 2 - half_l))
        bottom_right = (int(width / 2 + half_w), int(height / 2 + half_l))
        cv2.rectangle(image, top_left, bottom_right, (40, 40, 40), -1)
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1)

    def tick(self):
        if self.running:
            try:
                image, problems = self.stitch_once()
            except (ValueError, cv2.error) as exc:
                image, problems = None, [str(exc)]
            self.frame_count += 1
            if image is not None:
                self.last_output = image
                self.show(image)
            self.status.set(" | ".join(problems) if problems else (
                "Running." if image is not None else "Waiting for frames..."))
        if not self.closed:
            self.timer = self.root.after(33, self.tick)

    def show(self, image):
        width, height = max(self.view.winfo_width(), 1), max(self.view.winfo_height(), 1)
        picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        picture.thumbnail((width, height), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(picture)
        self.view.configure(image=photo, text="")
        self.view.image = photo

    def snapshot(self):
        if self.last_output is None:
            messagebox.showerror("Snapshot", "Nothing to save yet. Press Start first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path and not cv2.imwrite(path, self.last_output):
            messagebox.showerror("Snapshot", "Could not write the image.")

    def close(self):
        self.closed = True
        self.root.after_cancel(self.timer)
        self.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1000x900")
    StitchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
