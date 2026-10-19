# Path-following (orange path marker)

Vision + control for the following the path... detect the orange
marker on the pool floor with the bottom camera, measure how far we're off its
axis, n steer to stay on it.

UPDATE:
We now filter by orange instead of grayscale darkest colors. Thats about it

Slight, loose inspiration from ITU AUV (itu-auv/auv-software).
## Files


Hey chatgpt, explain my files

Sure thing! heres a table formatted in markdown!

| File | What it is |
|------|-----------|
| `opencv_pipe_detector.py` | camera image -> binary marker mask |
| `pipe_follower.py` | mask -> (forward speed, yaw rate) steering |
| `marker_angle.py` | how tilted the marker is (two methods) |
| `pipe_follower_params.yaml` | reference values for the follower gains |
| `testdetect.py` | quick script to eyeball the detector on an image |
| `test.png`, `test1.png` | sample images for testing |

## Run the detector test 

```bash
cd path_following
python testdetect.py
```
BTW I accidentally removed all comments from some files ill add them back later mb
## next

- Tune `PipeDetectorConfig` against real pool footage
- Confirm the two `marker_angle` methods agree, then use one
- Build the ROS2 things that use these.
- Break the 180 degree direction ambiguity (we dont know if we goin south or north) with IMU/compass
