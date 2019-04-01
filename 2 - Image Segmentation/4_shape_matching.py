import pandas as pd
import numpy as np
import cv2

# load with grayscale
template = cv2.imread('4star.jpg',0)
cv2.imshow('Original', template)
cv2.waitKey()

# Load target image
target = cv2.imread('shapestomatch.jpg')
cv2.imshow('Target', target)
cv2.waitKey()

# grayscale
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold both images
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# Find contours in template
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# we need to sort the contours by area so that we can remove the largest
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

# we extract the second largest contour which will be our template contour
template_contour = contours[1]

# Extract contours from second target image
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# THE MAGIC IS HERE
for c in contours:
    # Iterate through each contour in the target image and use cv2.matchShape() to compare contours
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    # if we match value is less than 0.15 we
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []

cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
