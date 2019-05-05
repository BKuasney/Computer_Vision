import pandas
import numpy as np
import cv2

# load as grayscale
# to see what this function make, use image2
image = cv2.imread('demo_5.jpeg',0)
image2 = cv2.imread('./images/gradient.jpg',0)

cv2.imshow('Original', image)
cv2.waitKey()

# Values below 80 goes to 0 (black) and above goes to 255 (white)
ret, thresh1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('1 - Threshold Binary', thresh1)
cv2.waitKey()

# inverse of THRESH_BINARY
# below 80 goes to 255 and above goes to 0
ret, thresh2 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('2 - Inverse Threshold Binary', thresh2)
cv2.waitKey()

# values above 80 are truncated at 80 (the 255 argument is unused)
ret, thresh3 = cv2.threshold(image, 80, 255, cv2.THRESH_TRUNC)
cv2.imshow('3 - Threshold Truncated', thresh3)
cv2.waitKey()

# Values below 80 goes to 0 (black), above are unchanged (remember image is grayscale)
ret, thresh4 = cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO)
cv2.imshow('4 - Threshold To Zero', thresh4)
cv2.waitKey()

# Inverse THRESH_TOZERO function
ret, thresh5 = cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('5 - Inverse Threshold To Zero', thresh5)
cv2.waitKey()


#######################################################################################################################
# the biggest downfall of those simple threshold methods is what we need to provide the threshold value (the 80 value)
# to make this more smarter we use ADAPTTIVE THRESHOLD
#######################################################################################################################

# First, its a good practice blur the images as it removes noise
image = cv2.GaussianBlur(image, (3,3), 0)

# Using AdaptiveThreshold
# arguments: image, max_value, adaptive_type, threshold_type, block_size, constant_that_is_subtracted_from_mean)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 3, 5)

cv2.imshow('Adaptive Mean Threshold', thresh)
cv2.waitKey()

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Otsu', th2)
cv2.waitKey()
