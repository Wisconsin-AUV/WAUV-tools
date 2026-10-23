"""Live odometry viz: JSON over UDP -> XY plot of current position.

Datagram (one JSON object each): {"x": 1.0, "y": 2.0, "z": 0.5}
(also accepts {"position": {"x":..., "y":..., "z":...}})

Usage: python3 odom_viz_udp.py --port 9870
"""
import argparse
import json
import socket

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=9870)
args = ap.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.host, args.port))
sock.setblocking(False)

xs, ys = [], []
cur = [0.0, 0.0, 0.0]  # current x, y, z

fig, ax = plt.subplots()
(trail,) = ax.plot([], [], "b-", lw=1)
(dot,) = ax.plot([], [], "ro")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True)


def update(_frame):
    while True:  # drain every datagram waiting on the socket
        try:
            data, _addr = sock.recvfrom(65535)
        except BlockingIOError:
            break
        try:
            msg = json.loads(data)
            p = msg.get("position", msg)
            cur[:] = float(p["x"]), float(p["y"]), float(p["z"])
        except (ValueError, KeyError, TypeError):
            continue  # ignore malformed packets
        xs.append(cur[0])
        ys.append(cur[1])

    trail.set_data(xs, ys)
    dot.set_data([cur[0]], [cur[1]])
    ax.set_title(f"x={cur[0]:.2f}  y={cur[1]:.2f}  z={cur[2]:.2f}")
    ax.relim()
    ax.autoscale_view()


anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()