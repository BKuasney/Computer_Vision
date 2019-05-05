import pandas as pd
import numpy as np
import cv2

image = cv2.imread('hand.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

# GrayScale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# colors between this range
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# sort Contoursby area and then remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key = cv2.contourArea, reverse=False)[:n]

# Iterate throught contours and draw the convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow('Convex Hull', image)

cv2.waitKey() 
