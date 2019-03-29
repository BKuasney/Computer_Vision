import pandas as pd
import numpy as np
import cv2

image = cv2.imread('./images/input.jpg',0)
image = cv2.imread('demo_5.jpeg',0)

height, width = image.shape


#############
# S O B E L #
#############

# Extract Sobel Edges
sobel_x = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize = 5)
sobel_y = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize = 5)

cv2.imshow('Original', image)
cv2.waitKey()
cv2.imshow('Sobel X', sobel_x)
cv2.waitKey()
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey()

# combine edges
sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('Sobel OR', sobel_OR)
cv2.waitKey()

#####################
# L A P L A C I A N #
#####################

# make eges more "clean"
laplacian = cv2.Laplacian(image, cv2.CV_8U)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey()


#############
# C A N N Y #
#############
# much better

# we can provide 2 values: threshold1 and threshold2
# any gradient value larger then threshold2 is considered to be an edge
# any gradient value below threshold1 is considered not to be an edge
# values between threshold1 and threshold2 are either classified as edges or non-edges based on how their intensities are "connected"
# in this case, any gradient values below 60 are considered non-edges
# whereas any value above 120 are considered edge

# much better
canny = cv2.Canny(image, 60, 120)
cv2.imshow('Canny', canny)
cv2.waitKey()
