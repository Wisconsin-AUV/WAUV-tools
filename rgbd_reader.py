import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # set before importing cv2

import cv2
import numpy as np

DATA_DIR = "../data"
FIRST, LAST = 1, 539

# cv2.waitKeyEx codes (arrow keys differ by platform); a/d work everywhere
SKIP = 50

# Arrows step 1 frame; a/d skip SKIP frames
LEFT_KEYS = {65361, 2424832, 63234}    # Linux, Windows, macOS
RIGHT_KEYS = {65363, 2555904, 63235}
SKIP_LEFT_KEYS = {ord("a")}
SKIP_RIGHT_KEYS = {ord("d")}
QUIT_KEYS = {27, ord("q")}                      # Esc or q

MAX_DEPTH = 8.0

def make_colorbar(height, width=70, bar_width=20, n_ticks=5):
    # Top = MAX_DEPTH, bottom = 0, same colormap as the depth image
    gradient = np.linspace(255, 0, height).astype(np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :bar_width] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)

    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        y = int((1 - frac) * (height - 1))
        cv2.line(canvas, (bar_width, y), (bar_width + 4, y), (255, 255, 255), 1)
        ty = min(max(y + 4, 12), height - 3)  # keep text inside the image
        cv2.putText(canvas, f"{MAX_DEPTH * frac:.0f}m", (bar_width + 7, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def load_frame(idx):
    path = os.path.join(DATA_DIR, f"out_rgbd_{idx}.exr")
    image = cv2.imread(
        path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
    )
    if image is None:
        return None

    rgb = image[:, :, :3].astype(np.uint8)
    depth = np.squeeze(image[:, :, -1:])
    depth = np.nan_to_num(depth, nan=MAX_DEPTH)
    depth = np.clip(depth, 0.0, MAX_DEPTH)
    depth = (depth / MAX_DEPTH * 255).astype(np.uint8)  # fixed scale, not per-frame

    rgb = rgb[::3, ::3]
    depth = depth[::3, ::3]
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    colorbar = make_colorbar(depth_color.shape[0])
    return np.hstack((depth_color, colorbar, rgb))


def render(idx):
    frame = load_frame(idx)
    if frame is None:
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        label = f"{idx}: file missing"
    else:
        label = f"{idx}/{LAST}"
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Depth (Left) vs RGB (Right)", frame)


idx = FIRST
render(idx)

while True:
    key = cv2.waitKeyEx(0)
    if key in QUIT_KEYS:
        break

    new_idx = idx
    if key in RIGHT_KEYS:
        new_idx = min(idx + 1, LAST)
    elif key in LEFT_KEYS:
        new_idx = max(idx - 1, FIRST)
    elif key in SKIP_RIGHT_KEYS:
        new_idx = min(idx + SKIP, LAST)
    elif key in SKIP_LEFT_KEYS:
        new_idx = max(idx - SKIP, FIRST)

    if new_idx != idx:
        idx = new_idx
        render(idx)

cv2.destroyAllWindows()