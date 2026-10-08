import socket
import cv2
import pyzed.sl as sl
import numpy as np
import os
import threading
import queue
import time

LAPTOP_ADDR = ("192.168.137.1", 5000)
STREAM_SIZE = (640, 360)           # size of the streamed image
JPEG_QUALITY = 70      # compression percentage, 100 means no compression

# Thread-safe queues to pass data between background threads
exr_queue = queue.Queue(maxsize=10)
mp4_queue = queue.Queue(maxsize=10)

# Event to cleanly shut down all background threads on Ctrl+C
stop_event = threading.Event()

zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA          # Set depth quality
init_params.coordinate_units = sl.UNIT.METER           # Set units for depth

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Camera Open Error: {err}")
    exit(1)

def save_unique_image(base_path, image):
    dirname, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)

    counter = 1
    unique_path = base_path

    while os.path.exists(unique_path):
        unique_path = os.path.join(dirname, f"{name}_{counter}{ext}")
        counter += 1

    cv2.imwrite(unique_path, image)
    print(f"Saved EXR: {unique_path}")

# BACKGROUND THREADS

def exr_saving_worker():
    """Background thread for writing EXR files"""
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)

    while not stop_event.is_set() or not exr_queue.empty():
        try:
            # Block for a brief moment waiting for data
            rgbd_image = exr_queue.get(timeout=1)
            save_unique_image("data/out_rgbd.exr", rgbd_image)
            exr_queue.task_done()
        except queue.Empty:
            continue

def mp4_saving_worker():
    """Background thread for the MP4 video"""
    rgb_output_file = "rgb_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(rgb_output_file, fourcc, 15.0, (1920, 1200))
    print(f"MP4 recording started -> {rgb_output_file}")

    while not stop_event.is_set() or not mp4_queue.empty():
        try:
            frame = mp4_queue.get(timeout=1)
            rgb_writer.write(frame)
            mp4_queue.task_done()
        except queue.Empty:
            continue

    rgb_writer.release()
    print("MP4 Writer cleanly released.")

# START THE THREADS

exr_thread = threading.Thread(target=exr_saving_worker, daemon=True)
mp4_thread = threading.Thread(target=mp4_saving_worker, daemon=True)

exr_thread.start()
mp4_thread.start()

# Initialize main thread UDP streamer
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
image = sl.Mat()
depth_image = sl.Mat()

runtime_params = sl.RuntimeParameters()

print("Streaming and recording initialized. Press Ctrl+C to stop.")

try:
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Grab camera frames immediately
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)

            # Deep copy frames because ZED sl.Mat reuse memory buffers
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
            depth_frame = depth_image.get_data().copy()

            # Hand off writing tasks to background threads
            if not mp4_queue.full():
                mp4_queue.put(frame.copy()) # copy to prevent thread frame modifications

            if not exr_queue.full():
                if len(depth_frame.shape) == 2:
                    depth_frame = np.expand_dims(depth_frame, axis=-1)
                rgbd_image = np.dstack((frame, depth_frame))
                exr_queue.put(rgbd_image)

            # Handle UDP Streaming instantly (No disk bottlenecks!)
            stream_frame = cv2.resize(frame, STREAM_SIZE, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            if len(buf) >= 65000:
                print(f"frame too big: {len(buf)} bytes, dropped")
                continue
            sock.sendto(buf.tobytes(), LAPTOP_ADDR)

except KeyboardInterrupt:
    print("\nStopping application and flushing remaining frames to disk...")
    # Signal workers to finish up what remains in their queues and close down
    stop_event.set()

    # Wait for the background queues to process their remaining items before exiting
    exr_thread.join()
    mp4_thread.join()

# Close SDK resources safely
zed.disable_recording()
zed.close()
sock.close()
print("All systems stopped cleanly.")