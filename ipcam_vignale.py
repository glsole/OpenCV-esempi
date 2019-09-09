import numpy as np
import cv2

source = "rtsp://admin:luca2006@79.12.36.85/Streaming/Channels/102"
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
# cv2.destroyAllWindows()
