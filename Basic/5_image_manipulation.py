import pandas as pd
import numpy as np
import cv2

#########################
# T R A N S L A T I O N #
#########################

image = cv2.imread('./images/input.jpg')
image = cv2.imread('demo_5.jpeg')

cv2.namedWindow('input', 0);
cv2.resizeWindow('input', 700,700);
cv2.imshow('input', image)
cv2.waitKey()

# store wight and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

# translation matrix
# | 1 0 Tx |
# | 0 1 Ty |

# T is our translation matrix
T = np.float32([[1,0, quarter_width], [0,1,quarter_height]])
print(T)
# we use warpAffine to transform the image using the matrix, T
# affine means keep proportionality
img_translation = cv2.warpAffine(image, T, (width, height))

#cv2.resizeWindow("Display frame", 50, 50)
cv2.namedWindow('Translation',0)
cv2.resizeWindow('Translation', 700,700)
cv2.imshow('Translation', img_translation)
cv2.waitKey()

#####################
# R O T A T I O N S #
#####################

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
# 90 degree
rotated_image = cv2.warpAffine(image, rotation_matrix,(width, height))

cv2.namedWindow('Rotation', 0)
cv2.resizeWindow('Rotation', 700,700)
cv2.imshow('Rotation', rotated_image)
cv2.waitKey()

# a simple way to rorate the image:
rotated_image = cv2.transpose(image)
cv2.imshow('Rotation', rotated_image)
cv2.waitKey()

#######################################
# SCALING / RESIZING / INTERPOLATIONS #
#######################################

# make our image 3/4 of it's original size
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
cv2.imshow('Scaling - Linear Interpolation', image_scaled)
cv2.waitKey()

# make our image double the size
image_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', image_scaled)
cv2.waitKey()

# make exact dimension
image_scaled = cv2.resize(image, (900,400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', image_scaled)
cv2.waitKey()

# sacling Pyramids: useful when scaling images in object detection
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller) # in relation of smaller size
larger2 = cv2.pyrUp(image) # in relation of original size

cv2.imshow('Smaller', smaller)
cv2.waitKey()
cv2.imshow('Larger', larger)
cv2.waitKey()
cv2.imshow('Larger2', larger2)
cv2.waitKey()

###################
# C R O P P I N G #
###################

# extract useful elements of image
image = cv2.imread('./images/input.jpg')
height, width = image.shape[:2]

# get the starting  pixel coordenates
start_row, start_col = int(height * .25), int(width * .25) # start in 25% of image
end_row, end_col = int(height * .75), int(width * .75) # ends in 75% of image

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow('Cropped', cropped)
cv2.waitKey()
