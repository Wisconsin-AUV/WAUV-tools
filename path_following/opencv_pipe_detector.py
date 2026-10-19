# Finds the path marker in a bottom-camera image and returns it as a black/white mask.
# Adapted from itu-auv/auv-software (BSD-3-Clause).

import cv2
import numpy as np
from dataclasses import dataclass


# Tunable settings for the detector
@dataclass
class PipeDetectorConfig:
    use_color: bool = True                 # True = find orange by color, False = find the darkest blob
    hsv_low: tuple = (15, 70, 45)          # lowest hue, saturation, brightness that counts as orange
    hsv_high: tuple = (50, 255, 255)       # highest hue, saturation, brightness that counts as orange
    use_percentile: bool = True            # dark-blob mode: pick the threshold from the image
    black_percentile: float = 5            # dark-blob mode: darkest 5% counts as marker
    fixed_thresh: int = 30                 # dark-blob mode: fixed threshold if use_percentile is False
    blur_ksize: int = 3                    # blur size to smooth noise
    morph_close: int = 7                   # fills small holes in the marker
    morph_open: int = 3                    # removes small specks
    min_area_px: int = 150                 # ignore blobs smaller than this


# Blur and morphology kernel sizes must be odd
def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else (k + 1)


# Orange mode: returns a black/white image where orange pixels are white (255)
def _orange_threshold(img_bgr, cfg):
    # YOUR CODE HERE 
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.hsv_low, cfg.hsv_high)
    return mask




# Dark-blob mode: returns a black/white image where the darkest pixels are white (255)
def _dark_threshold(img_bgr, cfg):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if cfg.blur_ksize and cfg.blur_ksize >= 3:
        k = _odd(cfg.blur_ksize)
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray

    if cfg.use_percentile:
        p = np.clip(cfg.black_percentile, 0, 100)
        thr_val = int(np.percentile(gray_blur, p))
        thr_val = max(thr_val, 10)
    else:
        thr_val = int(np.clip(cfg.fixed_thresh, 0, 255))

    _, thr = cv2.threshold(gray_blur, thr_val, 255, cv2.THRESH_BINARY_INV)
    return thr


# Takes a BGR image, returns (mask, thresh). mask only has the marker blob.
def detect_pipe_mask(img_bgr, cfg=PipeDetectorConfig()):
    # step 1: black/white image of the marker candidates, by color or by darkness
    if cfg.use_color:
        thr = _orange_threshold(img_bgr, cfg)
    else:
        thr = _dark_threshold(img_bgr, cfg)

    # step 2: clean up the edges: fill holes, then remove specks
    if cfg.morph_close and cfg.morph_close >= 3:
        k = _odd(cfg.morph_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    if cfg.morph_open and cfg.morph_open >= 3:
        k = _odd(cfg.morph_open)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    # step 3: keep only the biggest white blob and treat it as the marker
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    mask = np.zeros_like(thr)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        if stats[max_idx, cv2.CC_STAT_AREA] >= cfg.min_area_px:
            mask[labels == max_idx] = 255

    return mask, thr
