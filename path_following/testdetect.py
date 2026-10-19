# Quick test: runs the detector on an image and shows what it sees.
# Press any key to close the windows.

import cv2
from opencv_pipe_detector import detect_pipe_mask, PipeDetectorConfig

# load the test image (run this from inside path_following/)
img = cv2.imread("real7.png")
cfg = PipeDetectorConfig()

# thresh = black/white before cleanup, mask = final marker blob
mask, thresh = detect_pipe_mask(img, cfg)

cv2.imshow("original", img)
cv2.imshow("thresh", thresh)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
