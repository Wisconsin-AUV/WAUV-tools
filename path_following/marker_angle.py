# next : check that the two angle methods agree on real pool footage
# next : feed this angle into pipe_follower.py, then wrap it all in a ROS2 node

# Measures how tilted the marker is in the mask, two different ways.
# 0 degrees means the marker is straight up and down in the image.

import cv2
import numpy as np
import math


# biggest white blob in the mask, or None if the mask is empty
def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# Method 1: fit the tightest tilted rectangle around the marker and read its angle
def marker_angle_minrect(mask):
    cnt = _largest_contour(mask)
    if cnt is None:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(cnt)
    # opencv's angle is for whichever side it calls the width, so correct for that
    if w < h:
        norm = angle
    else:
        norm = angle + 90.0
    # wrap into -90 to 90
    while norm > 90.0:
        norm -= 180.0
    while norm <= -90.0:
        norm += 180.0
    return float(norm)


# Method 2: find the marker's center near the top and near the bottom, then measure the line between them
def marker_angle_two_point(mask):
    ys_all, xs_all = np.where(mask > 0)
    if len(ys_all)==0:
        return None
    y_min = ys_all.min()
    y_max = ys_all.max()
    band = max((y_max - y_min) // 4, 1)
    # top quarter and bottom quarter of the marker
    top = mask[y_min:y_min + band, :]
    bottom = mask[y_max - band + 1:y_max + 1, :]
    # average position of the white pixels in the top strip
    ys, xs, = np.where(top > 0)
    if len(xs) == 0:
        return None
    x_top = xs.mean()
    y_top = ys.mean() + y_min
    # same for the bottom strip
    ys, xs, = np.where(bottom > 0) #yes i am reusing the variable names
    x_bot = xs.mean()
    y_bot = ys.mean() + (y_max - band + 1)
    dx = x_top - x_bot
    dy = y_bot - y_top
    # angle of the line between the two points
    radians = math.atan2(dx, dy)
    degrees = math.degrees(radians) #i hate radians
    return degrees


# run this file directly to compare both methods on real7.png
if __name__ == "__main__":
    from opencv_pipe_detector import detect_pipe_mask, PipeDetectorConfig

    img = cv2.imread("real7.png")
    if img is None:
        print("Put real7.png next to this file.")
        raise SystemExit

    mask, _ = detect_pipe_mask(img, PipeDetectorConfig())
    print("minrect angle   :", marker_angle_minrect(mask))
    print("two-point angle :", marker_angle_two_point(mask))
