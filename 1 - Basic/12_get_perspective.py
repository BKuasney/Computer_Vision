import pandas as pd
import numpy as np
import cv2

image = cv2.imread('./images/scan.jpg')
cv2.imshow('Original', image)
cv2.waitKey()
quit()
# Coordinates of the 4 corners of the desired image
pointsA = np.float32([[320,15], [700,215], [85,610], [530, 780]])

# Coordinates of the 4 corners of the desired output
pointsB = np.float32([[0,0], [420,0], [0, 594], [420, 594]])

# use the two sets of four points to compute the perspective transformation matrix M
M = cv2.getPerspectiveTransform(pointsA, pointsB)

warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow('Perspective', warped)
cv2.waitKey()
