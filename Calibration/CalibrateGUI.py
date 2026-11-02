import base64
import math
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')  # quiet DSHOW warnings when probing missing camera indices
import cv2
import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calib')
MIN_FRAMES = 10
MAX_PROBE = 10
PREVIEW_RESOLUTIONS = [(640, 480), (320, 240)]
TILE_W, TILE_H = 480, 270
MAX_VIEW_W = 960
TICK_MS = 10


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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Intrinsic Calibration')
        self.mode = 'idle'  # idle | picker | calib
        self.photo = None
        self.pending_key = None
        self.sess = {}

        self.vars = {
            'names': tk.StringVar(),
            'cols': tk.StringVar(value='9'),
            'rows': tk.StringVar(value='6'),
            'square': tk.StringVar(value='25.0'),
            'width': tk.StringVar(value='1280'),
            'height': tk.StringVar(value='720'),
            'pinhole': tk.BooleanVar(value=False),
            'out': tk.StringVar(value=DEFAULT_DIR),
            'frames': tk.StringVar(value='20'),
            'interval': tk.StringVar(value='1.0'),
            'min_move': tk.StringVar(value='40.0'),
        }
        self.done = set()
        self._build()
        self.bind_all('<Key>', self._on_key)
        self.protocol('WM_DELETE_WINDOW', self._close)

    # ---------- layout ----------
    def _build(self):
        v = self.vars
        side = ttk.Frame(self)
        side.grid(row=0, column=0, sticky='ns', padx=6, pady=6)

        cams = ttk.LabelFrame(side, text='Cameras')
        cams.pack(fill='x', pady=3)
        self._row(cams, 0, 'Names (0=front 1=rear)', v['names'], 16)

        board = ttk.LabelFrame(side, text='Checkerboard')
        board.pack(fill='x', pady=3)
        self._row(board, 0, 'Inner corners (cols)', v['cols'])
        self._row(board, 1, 'Inner corners (rows)', v['rows'])
        self._row(board, 2, 'Square size (mm)', v['square'])

        cap = ttk.LabelFrame(side, text='Capture')
        cap.pack(fill='x', pady=3)
        self._row(cap, 0, 'Width', v['width'])
        self._row(cap, 1, 'Height', v['height'])
        self._row(cap, 2, 'Frames to auto-calibrate', v['frames'])
        self._row(cap, 3, 'Min seconds between saves', v['interval'])
        self._row(cap, 4, 'Min corner movement (px)', v['min_move'])
        ttk.Checkbutton(cap, text='Pinhole model (default fisheye)', variable=v['pinhole']).grid(
            row=5, column=0, columnspan=2, sticky='w', padx=6, pady=2)

        out = ttk.LabelFrame(side, text='Output folder')
        out.pack(fill='x', pady=3)
        ttk.Entry(out, textvariable=v['out'], width=24).grid(row=0, column=0, padx=6, pady=3)
        ttk.Button(out, text='...', width=3, command=self._browse).grid(row=0, column=1, padx=3)

        self.find_btn = ttk.Button(side, text='Find cameras', command=self._find_cameras)
        self.find_btn.pack(fill='x', pady=(8, 2))
        self.save_btn = ttk.Button(side, text='Save frame (Space)', command=lambda: self._press(' '),
                                   state='disabled')
        self.save_btn.pack(fill='x', pady=2)
        self.calib_btn = ttk.Button(side, text='Calibrate now (c)', command=lambda: self._press('c'),
                                    state='disabled')
        self.calib_btn.pack(fill='x', pady=2)
        self.back_btn = ttk.Button(side, text='Back / Stop (Backspace)', command=lambda: self._press('back'),
                                   state='disabled')
        self.back_btn.pack(fill='x', pady=2)

        main = ttk.Frame(self)
        main.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        self.view = tk.Label(main, bg='black', width=80, height=24, fg='white',
                             text='Click "Find cameras" to begin')
        self.view.grid(row=0, column=0, sticky='nsew')
        self.view.bind('<Button-1>', self._on_click)
        self.log = scrolledtext.ScrolledText(main, height=7, state='disabled')
        self.log.grid(row=1, column=0, sticky='ew', pady=(6, 0))

    @staticmethod
    def _row(parent, row, label, var, width=8):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=6, pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky='w', padx=6, pady=2)

    def _browse(self):
        path = filedialog.askdirectory(initialdir=self.vars['out'].get() or DEFAULT_DIR)
        if path:
            self.vars['out'].set(os.path.normpath(path))

    def _log(self, text):
        self.log.config(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.config(state='disabled')

    def _show(self, img):
        h, w = img.shape[:2]
        if w > MAX_VIEW_W:
            img = cv2.resize(img, (MAX_VIEW_W, int(h * MAX_VIEW_W / w)))
        ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
            self.view.config(image=self.photo, text='', width=0, height=0)

    def _reset_view(self, text):
        self.photo = None
        self.view.config(image='', text=text, width=80, height=24)

    def _set_mode(self, mode):
        self.mode = mode
        calib = mode == 'calib'
        self.find_btn.config(state='normal' if mode == 'idle' else 'disabled')
        self.back_btn.config(state='disabled' if mode == 'idle' else 'normal')
        self.save_btn.config(state='normal' if calib else 'disabled')
        self.calib_btn.config(state='normal' if calib else 'disabled')

    # ---------- settings ----------
    def _settings(self):
        v = {k: var.get() for k, var in self.vars.items()}
        names = {}
        for item in v['names'].split():
            index, _, name = item.partition('=')
            if not (index.isdigit() and name):
                raise ValueError(f"'{item}' should look like 0=front")
            names[int(index)] = name
        return {
            'names': names,
            'board': (int(v['cols']), int(v['rows'])),
            'square': float(v['square']),
            'width': int(v['width']),
            'height': int(v['height']),
            'pinhole': v['pinhole'],
            'out': v['out'].strip() or DEFAULT_DIR,
            'frames': int(v['frames']),
            'interval': float(v['interval']),
            'min_move': float(v['min_move']),
        }

    # ---------- input ----------
    def _press(self, key):
        self.pending_key = key

    def _on_key(self, event):
        if isinstance(self.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ('BackSpace', 'Escape'):
            self._press('back')
        elif event.char == ' ':
            self._press(' ')
        elif event.char and event.char.lower() in ('c', 'q'):
            self._press(event.char.lower())
        elif event.char and event.char.isdigit():
            self._press(event.char)

    def _on_click(self, event):
        self.view.focus_set()
        if self.mode == 'picker' and 'cols' in self.sess:
            col, row = event.x // TILE_W, event.y // TILE_H
            pos = row * self.sess['cols'] + col
            if col < self.sess['cols'] and 0 <= pos < len(self.sess['indices']):
                self._press(str(self.sess['indices'][pos]))

    # ---------- camera discovery / picker ----------
    def _find_cameras(self):
        try:
            self.settings = self._settings()
        except ValueError as e:
            messagebox.showerror('Invalid input', str(e))
            return
        self.find_btn.config(state='disabled')
        self._reset_view('Searching for cameras...')
        self.update_idletasks()
        indices = find_indices()
        if not indices:
            self._log('Error: no cameras found.')
            self._reset_view('No cameras found. Click "Find cameras" to retry.')
            self._set_mode('idle')
            return
        self.names = {i: f'cam{i}' for i in indices}
        self.names.update({i: n for i, n in self.settings['names'].items() if i in indices})
        self._log(f'Found cameras: {indices}')
        self.indices = indices
        self._start_picker()

    def _start_picker(self):
        caps = open_previews(self.indices)
        if not caps:
            self._log('Error: could not open the cameras for preview.')
            self._reset_view('Could not open cameras. Click "Find cameras" to retry.')
            self._set_mode('idle')
            return
        self.sess = {'caps': caps, 'indices': self.indices,
                     'cols': math.ceil(math.sqrt(len(self.indices)))}
        self.pending_key = None
        self._log("Click a camera (or press its number) to calibrate it.")
        self._set_mode('picker')
        self.after(TICK_MS, self._picker_tick)

    def _release_picker(self):
        for cap in self.sess.get('caps', {}).values():
            cap.release()
        self.sess = {}

    def _picker_tick(self):
        if self.mode != 'picker':
            return
        key, self.pending_key = self.pending_key, None
        if key in ('back', 'q'):
            self._release_picker()
            self._set_mode('idle')
            self._reset_view('Click "Find cameras" to begin')
            return
        if key and key.isdigit() and int(key) in self.indices:
            self._release_picker()
            self._start_calibration(int(key))
            return

        caps, cols = self.sess['caps'], self.sess['cols']
        tiles = []
        for i in self.indices:
            ret, frame = caps[i].read()
            tiles.append(make_tile(i, self.names[i], frame if ret else None, i in self.done))
        rows = math.ceil(len(tiles) / cols)
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        self._show(grid)
        self.after(TICK_MS, self._picker_tick)

    # ---------- calibration ----------
    def _start_calibration(self, index):
        s = self.settings
        board = s['board']
        template = np.zeros((board[0] * board[1], 3), np.float32)
        template[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * s['square']

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s['height'])
        if not cap.isOpened():
            self._log(f'Error: Could not open camera {index}.')
            cap.release()
            self._start_picker()
            return

        self.sess = {'cap': cap, 'index': index, 'name': self.names[index], 'template': template,
                     'obj': [], 'img': [], 'size': None, 'last_corners': None, 'last_save': 0.0}
        self.pending_key = None
        self._log(f"Calibrating '{self.names[index]}' (camera {index}). Move the board around; frames save "
                  f"automatically (target {s['frames']}).")
        self._set_mode('calib')
        self.after(TICK_MS, self._calib_tick)

    def _save_frame(self, corners):
        s, st = self.settings, self.sess
        st['obj'].append(st['template'].copy())
        st['img'].append(corners.reshape(-1, 1, 2).astype(np.float32 if s['pinhole'] else np.float64))
        self._log(f"Saved frame {len(st['img'])}")

    def _run_calibration(self):
        s, st = self.settings, self.sess
        if len(st['img']) < MIN_FRAMES:
            self._log(f"Need at least {MIN_FRAMES} frames, have {len(st['img'])}.")
            return False
        self._log('Calibrating...')
        self.update_idletasks()
        try:
            rms, K, D = calibrate(st['obj'], st['img'], st['size'], not s['pinhole'])
        except cv2.error as e:
            self._log(f'Calibration failed ({e}). Keep capturing more varied frames.')
            return False
        os.makedirs(s['out'], exist_ok=True)
        path = os.path.join(s['out'], f"{st['name']}.npz")
        np.savez(path, K=K, D=D, size=np.array(st['size']), fisheye=not s['pinhole'])
        self._log(f'RMS error: {rms:.3f}px. Saved to {path}')
        return True

    def _end_calibration(self, result):
        self.sess['cap'].release()
        index = self.sess['index']
        self.sess = {}
        if result == 'saved':
            self.done.add(index)
        if result == 'quit':
            self._set_mode('idle')
            self._reset_view('Click "Find cameras" to begin')
        else:
            self._start_picker()

    def _calib_tick(self):
        if self.mode != 'calib':
            return
        s, st = self.settings, self.sess
        key, self.pending_key = self.pending_key, None
        if key == 'q':
            self._end_calibration('quit')
            return
        if key == 'back':
            self._end_calibration('back')
            return

        ret, frame = st['cap'].read()
        if not ret:
            self._log("Error: Can't receive frame.")
            self._end_calibration('back')
            return
        size = st['size'] = (frame.shape[1], frame.shape[0])
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), s['board'])

        now = time.time()
        auto_saved = False
        if corners is not None and now - st['last_save'] >= s['interval'] and (
                st['last_corners'] is None
                or np.linalg.norm(corners - st['last_corners'], axis=2).mean() >= s['min_move']):
            self._save_frame(corners)
            st['last_corners'], st['last_save'], auto_saved = corners, now, True

        view = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(view, s['board'], corners, True)
        if auto_saved:
            cv2.rectangle(view, (0, 0), (size[0] - 1, size[1] - 1), (255, 255, 255), 12)
        status = f"{st['name']}: {len(st['img'])}/{s['frames']} frames" + ('' if corners is not None else ' (no board)')
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        self._show(view)

        if key == ' ' and corners is not None:
            self._save_frame(corners)
            st['last_corners'], st['last_save'] = corners, now
        elif key == 'c' and self._run_calibration():
            self._end_calibration('saved')
            return
        if len(st['img']) >= s['frames'] and auto_saved and self._run_calibration():
            self._end_calibration('saved')
            return
        self.after(TICK_MS, self._calib_tick)

    def _close(self):
        self.mode = 'idle'
        self._release_picker()
        if 'cap' in self.sess:
            self.sess['cap'].release()
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
