import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PipeDetectorConfig:
    use_percentile: bool = True
    black_percentile: float = 5
    fixed_thresh: int = 30
    blur_ksize: int = 3
    morph_close: int = 5
    morph_open: int = 3
    min_area_px: int = 1000


def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else (k + 1)


def detect_pipe_mask(img_bgr, cfg=PipeDetectorConfig()):
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

    if cfg.morph_close and cfg.morph_close >= 3:
        k = _odd(cfg.morph_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    if cfg.morph_open and cfg.morph_open >= 3:
        k = _odd(cfg.morph_open)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    mask = np.zeros_like(thr)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        if stats[max_idx, cv2.CC_STAT_AREA] >= cfg.min_area_px:
            mask[labels == max_idx] = 255

    return mask, thr
