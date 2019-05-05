import pandas as pd
import numpy as np
import cv2

import sys
print(sys.executable)


# Create body classifier
body_classifier  = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# video from a video file
cap = cv2.VideoCapture('walking.avi')

# Loop
while cap.isOpened():
    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrian', frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
