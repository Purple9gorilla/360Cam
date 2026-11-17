"""Tkinter interface for Windows fisheye camera calibration."""

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from pathlib import Path

from calibration import MIN_SAMPLES, calibrate, find_corners, load_calibration, make_maps, save_calibration
from camera_devices import discover_cameras

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


class CameraWorker(threading.Thread):
    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events
        self.commands = queue.Queue()
        self.frames = queue.Queue(maxsize=1)

    def run(self):
        capture = None
        token = 0
        try:
            while True:
                try:
                    try:
                        command, value, token = self.commands.get(timeout=0.02 if capture is None else 0)
                    except queue.Empty:
                        command = None
                    if command is not None and capture is not None:
                        capture.release()
                        capture = None
                    if command == "stop":
                        break
                    if command == "open":
                        capture = cv2.VideoCapture(value, cv2.CAP_DSHOW)
                        if not capture.isOpened():
                            capture.release()
                            capture = None
                            self.events.put(("camera_error", token, "Could not open the selected camera."))
                    if capture is None:
                        continue
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        capture.release()
                        capture = None
                        self.events.put(("camera_error", token, "Camera stopped delivering frames."))
                        continue
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.frames.put((token, frame))
                    time.sleep(0.01)
                except Exception as exc:
                    if capture is not None:
                        capture.release()
                        capture = None
                    self.events.put(("camera_error", token, f"Camera worker failed: {exc}"))
        finally:
            if capture is not None:
                capture.release()


class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Fisheye camera calibration")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.worker = CameraWorker(self.events)
        self.worker.start()
        self.cameras = []
        self.camera = None
        self.token = 0
        self.generation = 0
        self.frame = None
        self.frame_size = None
        self.samples = []
        self.calibration = None
        self.maps = None
        self.busy = False
        self.scanning = False
        self.closed = False
        self.camera_var = tk.StringVar()
        self.columns = tk.StringVar(value="9")
        self.rows = tk.StringVar(value="6")
        self.status = tk.StringVar(value="Scanning for cameras...")
        self.sample_status = tk.StringVar(value="Samples: 0 (10 minimum; 15+ recommended)")
        self.identity = tk.StringVar(value="No camera selected.")
        self.result = tk.StringVar(value="Undistorted preview unavailable until calibrated.")
        self._build()
        self.columns.trace_add("write", self.pattern_changed)
        self.rows.trace_add("write", self.pattern_changed)
        self.refresh()
        self.timer = self.root.after(30, self.poll)

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        controls = ttk.Frame(self.root, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Camera:").grid(row=0, column=0, sticky="w")
        self.dropdown = ttk.Combobox(controls, textvariable=self.camera_var, state="readonly", width=70)
        self.dropdown.grid(row=0, column=1, sticky="ew", padx=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.select_camera)
        ttk.Button(controls, text="Refresh", command=self.refresh).grid(row=0, column=2)
        ttk.Label(controls, textvariable=self.identity, wraplength=1100).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=4
        )
        options = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        options.grid(row=1, column=0, sticky="ew")
        ttk.Label(options, text="Checkerboard inner corners: columns").pack(side="left")
        ttk.Entry(options, textvariable=self.columns, width=4).pack(side="left", padx=4)
        ttk.Label(options, text="rows").pack(side="left")
        ttk.Entry(options, textvariable=self.rows, width=4).pack(side="left", padx=4)
        ttk.Button(options, text="Capture view", command=self.capture).pack(side="left", padx=4)
        ttk.Button(options, text="Reset samples", command=self.reset_samples).pack(side="left", padx=4)
        ttk.Button(options, text="Calibrate", command=self.start_calibration).pack(side="left", padx=4)
        ttk.Button(options, text="Save", command=self.save).pack(side="left", padx=4)
        ttk.Button(options, text="Load", command=self.load).pack(side="left", padx=4)
        info = ttk.Frame(self.root, padding=(8, 0))
        info.grid(row=2, column=0, sticky="ew")
        ttk.Label(info, textvariable=self.sample_status).pack(anchor="w")
        ttk.Label(info, textvariable=self.result).pack(anchor="w")
        preview = ttk.Frame(self.root, padding=8)
        preview.grid(row=3, column=0, sticky="nsew")
        preview.columnconfigure((0, 1), weight=1, uniform="preview")
        preview.rowconfigure(1, weight=1)
        ttk.Label(preview, text="Distorted / original").grid(row=0, column=0)
        ttk.Label(preview, text="Undistorted").grid(row=0, column=1)
        self.original = ttk.Label(preview, anchor="center", text="No camera frame")
        self.corrected = ttk.Label(preview, anchor="center", text="Not calibrated")
        self.original.grid(row=1, column=0, sticky="nsew")
        self.corrected.grid(row=1, column=1, sticky="nsew")
        ttk.Label(self.root, textvariable=self.status, padding=8, wraplength=1100).grid(
            row=4, column=0, sticky="ew"
        )

    def run_job(self, kind, token, function):
        def work():
            try:
                self.events.put((kind, token, function(), None))
            except Exception as exc:
                self.events.put((kind, token, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def refresh(self):
        if self.scanning:
            return
        self.scanning = True
        self.token += 1
        self.worker.commands.put(("close", None, self.token))
        self.camera = None
        self.frame = None
        self.frame_size = None
        self.camera_var.set("")
        self.dropdown["values"] = ()
        self.identity.set("No camera selected.")
        self.clear_state()
        self.clear_previews()
        self.status.set("Scanning for Windows cameras...")
        self.run_job("discovery", self.token, discover_cameras)

    def select_camera(self, _event=None):
        index = self.dropdown.current()
        if index < 0:
            return
        self.token += 1
        self.camera = self.cameras[index]
        self.frame = None
        self.frame_size = None
        self.clear_state()
        self.clear_previews()
        self.identity.set(
            f"Enumerated device instance: {self.camera.instance_id or 'unavailable'} | "
            f"Hardware IDs: {', '.join(self.camera.hardware_ids) or 'unavailable'} | "
            "Index-to-device mapping is assumed, not verified. Refresh and visually verify."
        )
        self.status.set("Opening camera...")
        self.worker.commands.put(("open", self.camera.index, self.token))

    def pattern(self):
        try:
            pattern = (int(self.columns.get()), int(self.rows.get()))
        except ValueError as exc:
            raise ValueError("Enter whole numbers for checkerboard inner corners.") from exc
        if min(pattern) < 2 or max(pattern) > 100:
            raise ValueError("Checkerboard inner-corner counts must be between 2 and 100.")
        return pattern

    def pattern_changed(self, *_):
        self.clear_state()
        self.status.set("Checkerboard settings changed; capture new samples.")

    def clear_state(self):
        self.generation += 1
        self.samples = []
        self.calibration = None
        self.maps = None
        self.busy = False
        self.sample_status.set(f"Samples: 0 ({MIN_SAMPLES} minimum; 15+ recommended)")
        self.result.set("Undistorted preview unavailable until calibrated.")
        self.corrected.configure(image="", text="Not calibrated")
        self.corrected.image = None

    def reset_samples(self):
        self.clear_state()
        self.status.set("Samples and calibration cleared.")

    def clear_previews(self):
        for label, text in ((self.original, "No camera frame"), (self.corrected, "Not calibrated")):
            label.configure(image="", text=text)
            label.image = None

    def capture(self):
        if self.busy:
            return
        try:
            pattern = self.pattern()
            if self.frame is None:
                raise ValueError("Select an active camera before capturing a view.")
        except ValueError as exc:
            messagebox.showerror("Capture", str(exc))
            return
        frame = self.frame.copy()
        self.busy = True
        self.status.set("Finding checkerboard corners...")
        self.run_job("capture", self.generation, lambda: find_corners(frame, pattern))

    def start_calibration(self):
        if self.busy:
            return
        try:
            pattern = self.pattern()
            if self.frame_size is None or self.camera is None:
                raise ValueError("Select an active camera first.")
            if len(self.samples) < MIN_SAMPLES:
                raise ValueError(f"Capture at least {MIN_SAMPLES} valid views first.")
        except ValueError as exc:
            messagebox.showerror("Calibration", str(exc))
            return
        samples = list(self.samples)
        size = self.frame_size
        camera = self.camera
        self.busy = True
        self.status.set("Calibrating; please wait...")
        self.run_job("calibration", self.generation, lambda: calibrate(
            samples, pattern, size, camera.device_path, camera.instance_id
        ))

    def save(self):
        if self.calibration is None:
            messagebox.showerror("Save calibration", "Calibrate or load a calibration first.")
            return
        CALIB_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=CALIB_DIR, initialfile="calibration.json",
            defaultextension=".json", filetypes=[("JSON", "*.json")],
        )
        if path:
            try:
                save_calibration(path, self.calibration)
                self.status.set(f"Saved calibration to {path}")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Save calibration", str(exc))

    def load(self):
        if self.camera is None or self.frame_size is None:
            messagebox.showerror("Load calibration", "Select an active camera first.")
            return
        CALIB_DIR.mkdir(exist_ok=True)
        path = filedialog.askopenfilename(initialdir=CALIB_DIR, filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            value = load_calibration(path)
            if value.size != self.frame_size:
                raise ValueError(f"Calibration resolution {value.size} does not match camera {self.frame_size}.")
            if value.device_path and self.camera.device_path:
                if value.device_path.casefold() != self.camera.device_path.casefold():
                    raise ValueError("This calibration belongs to a different camera device path.")
            elif not messagebox.askyesno(
                "Camera identity unavailable",
                "The saved or current camera has no device path. Cannot verify identity. Apply anyway?"
            ):
                return
            maps = make_maps(value)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Load calibration", str(exc))
            return
        self.columns.set(str(value.pattern[0]))
        self.rows.set(str(value.pattern[1]))
        self.calibration = value
        self.maps = maps
        self.result.set(f"Calibration RMS reprojection error: {value.rms:.3f} pixels")
        self.status.set("Calibration loaded. Verify that the selected camera is correct.")

    def show_image(self, label, frame):
        width = max(label.winfo_width(), 1)
        height = max(label.winfo_height(), 1)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        label.image = photo

    def poll(self):
        try:
            while True:
                self.handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        try:
            token, frame = self.worker.frames.get_nowait()
            if token == self.token and self.camera is not None:
                size = (frame.shape[1], frame.shape[0])
                if self.frame_size is not None and size != self.frame_size:
                    self.clear_state()
                    self.status.set("Camera resolution changed; calibration and samples cleared.")
                self.frame_size = size
                self.frame = frame
                if self.status.get() == "Opening camera...":
                    self.status.set("Camera ready. Position the checkerboard and capture a view.")
                self.show_image(self.original, frame)
                if self.maps is not None:
                    self.show_image(self.corrected, cv2.remap(frame, *self.maps, interpolation=cv2.INTER_LINEAR))
        except queue.Empty:
            pass
        if not self.closed:
            self.timer = self.root.after(30, self.poll)

    def handle_event(self, event):
        kind, token, *payload = event
        if kind == "camera_error":
            if token == self.token:
                self.frame = None
                self.frame_size = None
                self.clear_state()
                self.clear_previews()
                self.status.set(payload[0])
                messagebox.showerror("Camera", payload[0])
            return
        value, error = payload
        if kind == "discovery":
            self.scanning = False
            if token != self.token:
                return
            if error:
                self.status.set(f"Camera discovery failed: {error}")
                return
            self.cameras, warning = value
            self.dropdown["values"] = [camera.label for camera in self.cameras]
            self.status.set((warning + " " if warning else "") + (
                "Select a camera. Index-to-device mapping is assumed, not verified."
                if self.cameras else "No cameras found; connect a camera and refresh."
            ))
            return
        if token != self.generation:
            return
        self.busy = False
        if error:
            self.status.set(f"{kind.capitalize()} failed: {error}")
            messagebox.showerror(kind.capitalize(), error)
        elif kind == "capture":
            if value is None:
                self.status.set("Checkerboard not detected. Ensure the entire board is visible and try again.")
            else:
                self.samples.append(value)
                self.sample_status.set(f"Samples: {len(self.samples)} ({MIN_SAMPLES} minimum; 15+ recommended)")
                self.status.set("View captured. Move and tilt the board for the next view.")
        elif kind == "calibration":
            try:
                maps = make_maps(value)
            except (ValueError, cv2.error) as exc:
                self.status.set(f"Could not create undistortion maps: {exc}")
                messagebox.showerror("Calibration", str(exc))
                return
            self.calibration = value
            self.maps = maps
            self.result.set(f"Calibration RMS reprojection error: {value.rms:.3f} pixels")
            self.status.set("Calibration complete. Compare the previews and save the result.")

    def close(self):
        self.closed = True
        self.root.after_cancel(self.timer)
        self.worker.commands.put(("stop", None, self.token))
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1200x700")
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
