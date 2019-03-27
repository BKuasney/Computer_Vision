import pandas as pd
import numpy as np
import cv2

# Create a black image
image = np.zeros((512,512,3), np.uint8)

# make in black and white
image2 = np.zeros((512,512), np.uint8)

cv2.imshow('Black', image)
cv2.waitKey()
cv2.imshow('Black and White', image2)
cv2.waitKey()


# let's draw a line over our black square

# Draw a diagonal blue line of thickness of 5 pixels
image = np.zeros((512,512,3), np.uint8)
cv2.line(image, (0,0), (511, 511), (255,127,0), 5)
cv2.imshow('Line', image)
cv2.waitKey()

# Draw a rectangle
cv2.rectangle(image, (100,100), (300, 250), (127,50,127), 5)
cv2.imshow('Rectangle', image)
cv2.waitKey()

# to fill object:
cv2.rectangle(image, (100,100), (300, 250), (127,50,127), -1)
cv2.imshow('Rectangle', image)
cv2.waitKey()

# make a cyrcle
cv2.circle(image, (350,350), 100, (15, 75,50), -1)
cv2.imshow('Circle', image)
cv2.waitKey()

# Polygons
image = np.zeros((512,512,3), np.uint8)
# define 4 points
pts = np.array([[10,50], [400,50], [90,200], [50,500]], np.int32)
# reshape
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, (0,0,255), 3)
cv2.resize(image, (960, 540))
cv2.imshow('Polygon', image)
cv2.waitKey()


# Add texts into a image
# at FONT_HERSHEY_COMPLEX local we can put other fonts
image = np.zeros((512,512,3), np.uint8)
cv2.putText(image, 'Hello World', (75, 290), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 170, 0), 3)
cv2.imshow('Hello World', image)
cv2.waitKey()
