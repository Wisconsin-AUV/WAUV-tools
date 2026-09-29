# next : check that the two angle methods agree on real pool footage
# next : feed this angle into pipe_follower.py, then wrap it all in a ROS2 node

import cv2
import numpy as np
import math


def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def marker_angle_minrect(mask):
    cnt = _largest_contour(mask)
    if cnt is None:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:
        norm = angle
    else:
        norm = angle + 90.0
    while norm > 90.0:
        norm -= 180.0
    while norm <= -90.0:
        norm += 180.0
    return float(norm)


def marker_angle_two_point(mask):
    h = mask.shape[0]
    band = h // 4
    top = mask[0:band, :]
    bottom = mask[h-band:h, :]
    ys, xs, = np.where(top > 0)
    if len(xs) == 0:
        return None
    x_top = xs.mean()
    y_top = ys.mean()
    ys, xs, = np.where(bottom > 0) #yes i am reusing the variable names
    x_bot = xs.mean()
    y_bot = ys.mean()
    dx = x_top - x_bot
    dy = y_top - y_bot
    radians = math.atan2(dx, dy)
    degrees = math.degrees(radians) #i hate radians
    return degrees


if __name__ == "__main__":
    from opencv_pipe_detector import detect_pipe_mask, PipeDetectorConfig

    img = cv2.imread("test1.png")
    if img is None:
        print("Put a test image named test1.png next to this file.")
        raise SystemExit

    mask, _ = detect_pipe_mask(img, PipeDetectorConfig())
    print("minrect angle   :", marker_angle_minrect(mask))
    print("two-point angle :", marker_angle_two_point(mask))
