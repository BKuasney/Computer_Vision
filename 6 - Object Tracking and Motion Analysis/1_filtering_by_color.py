import pandas as pd
import numpy as np
import cv2

# initialize Webcam
cap = cv2.VideoCapture(0)

# define range od blue color in  H S V
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Loop until break statement is executed
while True:

    # read webcam
    ret, frame = cap.read()

    # convert image from RGB/BRG to HSV so we easily filter
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # use inRange to capture only the values between lower and upper purple
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    # Perform Bitwise AND mask our original frame
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('Original', frame)

    cv2.imshow('mask', mask)
    cv2.imshow('Filtered by Color Only', res)
    if cv2.waitKey(1) == 13:
        break
