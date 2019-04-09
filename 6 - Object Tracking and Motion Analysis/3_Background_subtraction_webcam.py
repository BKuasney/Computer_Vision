# OpenCV 2.4.13 only
import numpy as np
import cv2

# Intialize Webcam
cap = cv2.VideoCapture(0)

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

while True:
    ret, frame = cap.read()

    # Apply background subtractor to get our foreground mask
    foreground_mask = foreground_background.apply(frame)

    cv2.imshow('Output', foreground_mask)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

'''
    Let's improve
'''

# Intialize Webcam
cap = cv2.VideoCapture(0)

# Initlaize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()

while True:
    ret, frame = cap.read()

    # Apply background subtractor to get our foreground mask
    foreground_mask = foreground_background.apply(frame)

    cv2.imshow('Output', foreground_mask)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
