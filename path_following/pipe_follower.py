import math
import numpy as np
import cv2
from dataclasses import dataclass


def saturate(x, lo, hi):
    return max(lo, min(hi, x))


def rot90n(img, n_cw):
    n_cw = int(n_cw) % 4
    if n_cw == 0:
        return img
    elif n_cw == 1:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif n_cw == 2:
        return np.ascontiguousarray(np.rot90(img, k=2))
    else:
        return np.ascontiguousarray(np.rot90(img, k=1))


@dataclass
class PipeFollowerConfig:
    rotate_cw: int = 0
    flip_x: bool = False
    flip_y: bool = False
    n_sensors: int = 5
    sensor_x_frac: float = 0.20
    sensor_width_frac: float = 0.10
    sensor_min_fill: float = 0.02
    k_y: float = 1.50
    k_ang: float = 1.00
    v_fwd: float = 0.25
    v_min: float = 0.05
    ang_limit: float = 0.6
    lin_limit: float = 0.4
    slowdown_angle: float = 0.8
    invert_y_error: bool = False
    invert_ang_error: bool = False
    min_pipe_area_px: int = 1500
    search_omega: float = 0.25


class PipeLineFollower(object):
    def __init__(self, cfg=PipeFollowerConfig()):
        self.p = cfg

    def compute_cmd(self, mask):
        p = self.p

        if p.rotate_cw:
            mask = rot90n(mask, p.rotate_cw)
        if p.flip_x:
            mask = np.ascontiguousarray(np.fliplr(mask))
        if p.flip_y:
            mask = np.ascontiguousarray(np.flipud(mask))

        h, w = mask.shape[:2]

        pipe_area = int(np.count_nonzero(mask))
        have_pipe = pipe_area >= p.min_pipe_area_px
        if not have_pipe:
            w_z = saturate(p.search_omega, -p.ang_limit, p.ang_limit)
            return 0.0, w_z, {"note": "SEARCH", "fit": None, "gate": None}

        _ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(_ret) == 3:
            _img_out, contours, _hier = _ret
        else:
            contours, _hier = _ret
        contours = list(contours) if contours is not None else []
        if len(contours) == 0:
            w_z = saturate(p.search_omega, -p.ang_limit, p.ang_limit)
            return 0.0, w_z, {"note": "SEARCH(no-contours)", "fit": None, "gate": None}

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pts = contours[0].reshape(-1, 2)

        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = math.atan2(float(vy), float(vx))
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        ang_err = angle
        if p.invert_ang_error:
            ang_err = -ang_err

        x_center = int(saturate(p.sensor_x_frac, 0.0, 1.0) * w)
        half_gate = int(0.5 * saturate(p.sensor_width_frac, 0.01, 0.9) * w)
        xs = max(0, x_center - half_gate)
        xe = min(w, x_center + half_gate)
        gate = mask[:, xs:xe]

        n = max(3, int(p.n_sensors))
        band_h = h // n
        occ = []
        centers_y = []
        for i in range(n):
            ys = i * band_h
            ye = h if i == n - 1 else (i + 1) * band_h
            roi = gate[ys:ye, :]
            fill = float(np.count_nonzero(roi)) / float(roi.size)
            occ.append(fill)
            centers_y.append(0.5 * (ys + ye))
        occ = np.array(occ, dtype=np.float32)
        centers_y = np.array(centers_y, dtype=np.float32)

        mask_valid = occ >= p.sensor_min_fill
        if np.any(mask_valid):
            y_meas = np.average(centers_y[mask_valid], weights=occ[mask_valid])
        else:
            m = cv2.moments(mask, binaryImage=True)
            if m["m00"] > 1e-3:
                y_meas = m["m01"] / m["m00"]
            else:
                y_meas = h * 0.5

        y_err = (y_meas - (0.5 * h)) / (0.5 * h)
        if p.invert_y_error:
            y_err = -y_err

        w_z = -p.k_ang * ang_err - p.k_y * y_err
        w_z = saturate(w_z, -p.ang_limit, p.ang_limit)

        speed_scale = 1.0 / (1.0 + p.slowdown_angle * abs(ang_err))
        v_x = saturate(p.v_fwd * speed_scale, p.v_min, p.lin_limit)

        info = {
            "note": "FOLLOW",
            "fit": (float(vx), float(vy), float(x0), float(y0)),
            "gate": (xs, xe),
            "y_meas": float(y_meas),
            "ang_err": float(ang_err),
            "y_err": float(y_err),
            "occ": occ.tolist(),
        }
        return v_x, w_z, info

    @staticmethod
    def draw_debug(mask, info):
        h, w = mask.shape[:2]
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        gate = info.get("gate")
        if gate is not None:
            xs, xe = gate
            cv2.rectangle(vis, (xs, 0), (xe, h - 1), (0, 255, 255), 1)
        fit = info.get("fit")
        if fit is not None:
            vx, vy, x0, y0 = fit

            def line_point(t):
                return int(x0 + t * vx), int(y0 + t * vy)

            cv2.line(vis, line_point(-2000), line_point(2000), (0, 0, 255), 2)
        cv2.line(vis, (0, int(h / 2)), (w - 1, int(h / 2)), (255, 0, 0), 1)
        y_meas = info.get("y_meas")
        if y_meas is not None:
            y = int(y_meas)
            cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 1)
        note = info.get("note")
        if note:
            cv2.putText(vis, str(note), (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2, cv2.LINE_AA)
        if "ang_err" in info:
            txt = "ang_err={:+.2f}  y_err={:+.2f}".format(
                info.get("ang_err", 0.0), info.get("y_err", 0.0))
            cv2.putText(vis, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (50, 220, 50), 2, cv2.LINE_AA)
        return vis
