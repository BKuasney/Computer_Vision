import pandas as pd
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# define range yellow
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# empty points array
points = []

# default camera window size
ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 2

while True:
    # Capture webcam frame
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold the HSV image to get only yellow images
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    # Find contours, openCV3
    #_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create empty center array to store centroid center of mass
    center = int(Height/2), int(Width/2)

    if len(contours) > 0:
        # Get the largest contour and its center
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:
            center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

        except:
            center = int(Height/2), int(Width/2)

        # Allow only contours that have a larger than 15 pixels
        if radius > 15:
            # Draw circle and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    # Log center points
    points.append(center)

    # loop over the set of tracked points
    if radius > 15:
        for i in range(1, len(points)):
            try:
                cv2.line(frame, points[i + 1], points[i], (0, 255, 0),2)
            except:
                pass

        # Make frame count zero
        frame_count = 0

    else:
        # Count frames
        frame_count += 1

        # if we count 10 frames without objects lets delete our trail
        if frame_count == 10:
            points = []
            # when frame_count reaches 20 let's clear our trail
            frame_count = 0

    # Display our object tracker
    frame = cv2.flip(frame, 1)
    cv2.imshow('Object Tracker', frame)

    if cv2.waitKey(1) == 13:
        break
