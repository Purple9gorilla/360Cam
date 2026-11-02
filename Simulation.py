"""Synthetic cameras around a virtual car, used to test the bird's-eye pipeline without hardware."""
import numpy as np
import cv2

from Calibration import CalibrateGround as G

TILE_MM = 500
PALETTE = np.array([[200, 80, 60], [80, 200, 90], [70, 90, 210], [200, 190, 70]], dtype=np.float64)


def floor_color(x, y):
    """Floor texture defined in ground coordinates, so every camera sees the same floor."""
    i = np.floor(x / TILE_MM).astype(int)
    j = np.floor(y / TILE_MM).astype(int)
    color = PALETTE[(i % 2) + 2 * (j % 2)] * (0.6 + 0.4 * ((i + j) % 2))[:, None]
    fx, fy = np.mod(x, TILE_MM), np.mod(y, TILE_MM)
    line = (fx < 15) | (fy < 15)
    color[line] = 20
    return color.astype(np.uint8)


def rotation(yaw_deg, pitch_deg):
    """World-to-camera rotation. World: x right, y forward, z up. Yaw 0 faces +y, positive turns left."""
    th, ph = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([-np.cos(ph) * np.sin(th), np.cos(ph) * np.cos(th), -np.sin(ph)])
    r = np.array([np.cos(th), np.sin(th), 0.0])
    return np.vstack([r, np.cross(f, r), f])


def render_frame(K, D, R, pos, size):
    u, v = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pix, K, D).reshape(-1, 2)
    rays = np.hstack([norm, np.ones((len(norm), 1))]) @ R
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -pos[2] / rays[:, 2]
    valid = (rays[:, 2] < -1e-6) & (t > 0) & (t < 30000)
    frame = np.full((len(norm), 3), 40, dtype=np.uint8)
    ground = pos[:2] + t[valid, None] * rays[valid, :2]
    frame[valid] = floor_color(ground[:, 0], ground[:, 1])
    return frame.reshape(size[1], size[0], 3)


def project_ground(K, D, R, pos, pts_mm):
    world = np.hstack([pts_mm, np.zeros((len(pts_mm), 1))])
    cam = (world - pos) @ R.T
    pix, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return pix.reshape(-1, 2)


def render_truth(canvas, scale):
    xs, ys = np.meshgrid(np.arange(canvas[0]), np.arange(canvas[1]))
    x = (xs.ravel() - canvas[0] / 2) / scale
    y = (canvas[1] / 2 - ys.ravel()) / scale
    return floor_color(x, y).reshape(canvas[1], canvas[0], 3)


def make_cameras(canvas=(1200, 1200), scale=0.2, size=(1280, 720), car_mm=(1800, 4500), height=700, pitch=45,
                 focal=540.0):
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    D = np.array([-0.1, 0.02, 0, 0, 0.0])
    half_w, half_l = car_mm[0] / 2, car_mm[1] / 2
    layout = {'front': ((0, half_l), 0), 'left': ((-half_w, 0), 90),
              'rear': ((0, -half_l), 180), 'right': ((half_w, 0), -90)}
    cams = []
    for name, (xy, yaw) in layout.items():
        pos = np.array([xy[0], xy[1], height], dtype=np.float64)
        R = rotation(yaw, pitch)
        th = np.radians(yaw)
        fwd, rgt = np.array([-np.sin(th), np.cos(th)]), np.array([np.cos(th), np.sin(th)])
        g = np.array([pos[:2] + d * fwd + s * rgt
                      for d in np.linspace(600, 2800, 8) for s in np.linspace(-1200, 1200, 7)])
        px = project_ground(K, D, R, pos, g)
        ok = (px[:, 0] > 0) & (px[:, 0] < size[0]) & (px[:, 1] > 0) & (px[:, 1] < size[1])
        H = G.compute_homography(px[ok], g[ok], K, D, False, scale, canvas)
        cams.append({'name': name, 'K': K, 'D': D, 'fisheye': False, 'H': H,
                     'frame': render_frame(K, D, R, pos, size)})
    return cams
