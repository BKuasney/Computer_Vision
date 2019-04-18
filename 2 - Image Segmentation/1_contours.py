import pandas as pd
import numpy as np
import cv2


# import a simple image with 3 black squares
image = cv2.imread('shapes.jpg')
image = cv2.imread('shapes_donut.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey()

# findíng contours
# use a copy of your image e.g edged.copy() since findContours alters the image
# findContours: image, Retrivial Mode, Approximation method
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# APPROXIMATION METHODS
# CHAIN_APPROX_NONE: stores all the boundary points. But, we don't necessaraly nedd all.
# if the points form a straight line, we only need the start and ending points
# CHAIN_APPROX_SIMPLE: instead only provides these start and end points of boundary contours, thus resulting in much more efficient storage contour information
# HIERARCHY ON CONTOUR
# cv2.RETR_LIST: Retrieves all contours
# cv2.RETR_EXTERNAL: Retrieves all in a 2-level hierarchy
# cv2.RETR_COMP: Retrieves all in a 2-level hierarchy
# cv2.RETR_TREE: Retrieves all in full hierarchy



cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey()

# OpenCV store contours in a list of lists
print(contours)
print('Number of Contours found: {}'.format(str(len(contours))))

# Draw all contours
# use -1 as the 3rd parameter to draw all
# 4th argument is the color
# 5th argument is the quantitity of contours
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', image)
cv2.waitKey()
