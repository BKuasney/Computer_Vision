import pandas as pd
import numpy as np
import cv2

image = cv2.imread('demo_5.jpeg')
#image = cv2.imread('./images/opencv_inv.png')

# Define kernel size
kernel = np.ones((5,5), np.uint8)

# erode
erosion = cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey()

# dilation
dilation = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv2.waitKey()

# opening - good for remove noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey()

# closing - good for remove noise
close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', close)
cv2.waitKey()
