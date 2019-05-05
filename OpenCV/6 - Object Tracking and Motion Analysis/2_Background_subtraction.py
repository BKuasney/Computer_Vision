"""
    GAUSSIAN MIXTURE-BASED BACKGROUND/FOREGROUND SEGMENTATION ALGORITHM
"""

import numpy as np
import pandas as pd
import cv2

cap = cv2.VideoCapture('walking.avi')

# initialize background subtractor
foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()

    # Apply background subtractor to get our foreground mask
    foreground_mask = foreground_background.apply(frame)

    cv2.imshow('Output', foreground_mask)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
