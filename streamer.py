import socket
import numpy as np
import cv2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5000))
print("Waiting for frames on port 5000...")
try:
    while True:
        data, addr = sock.recvfrom(65507)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        cv2.imshow("ZED RGB feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("\nStopping streaming...")