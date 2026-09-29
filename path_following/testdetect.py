import cv2
from opencv_pipe_detector import detect_pipe_mask, PipeDetectorConfig

img = cv2.imread("test1.png")
cfg = PipeDetectorConfig()

mask, thresh = detect_pipe_mask(img, cfg)

cv2.imshow("original", img)
cv2.imshow("thresh", thresh)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()