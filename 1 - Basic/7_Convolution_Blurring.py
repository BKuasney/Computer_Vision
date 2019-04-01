import pandas as pd
import numpy as np
import cv2

image = cv2.imread('demo_5.jpeg')
cv2.imshow('original', image)
cv2.waitKey()
# Creating 3x3 kernel
kernel_3x3 = np.ones((3,3), np.float32) / 9 #(3*3) # matrix with 0.111

# use  the cv2.filter2D to convolve the kernel with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred)
cv2.waitKey()

# creating kernel 7x7
kernel_7x7 = np.ones((7,7), np.float32) / 49 # (7*7) # matrix with 0.111
blurred = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred)
cv2.waitKey()


# Commom used blurring methods

# averaging
# use a normalized box filter
# this takes the pixels under the box and replaces the central element
# box to take a avg
blur = cv2.blur(image, (3,3)) # kernel 3x3
cv2.imshow('Averaging', blur)
cv2.waitKey()

# instead box filter, use a gaussian kernel
# gaussian uses a gaussian window (more emphasis on point around the kernel center)
gaussian0 = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian Kernel', gaussian0)
cv2.waitKey(0)
gaussian1 = cv2.GaussianBlur(image, (7,7), 1)
cv2.imshow('Gaussian Kernel', gaussian1)
cv2.waitKey(0)
gaussian2 = cv2.GaussianBlur(image, (7,7), 2)
cv2.imshow('Gaussian Kernel', gaussian2)
cv2.waitKey(0)

# Takes median of all the pixels under kernel area and central
# element is replaced with this median value
# use median of all elements
median = cv2.medianBlur(image, 5)
cv2.imshow('Median', median)
cv2.waitKey()

# Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral kernel 75', bilateral)
cv2.waitKey()

bilateral = cv2.bilateralFilter(image, 9, 20, 20)
cv2.imshow('Bilateral kernel 20', bilateral)
cv2.waitKey()

bilateral = cv2.bilateralFilter(image, 9, 5, 5)
cv2.imshow('Bilateral kernel 5', bilateral)
cv2.waitKey()

bilateral = cv2.bilateralFilter(image, 9, 100, 100)
cv2.imshow('Bilateral kernel 100', bilateral)
cv2.waitKey()


# Image De-Noise - Non-Local Means Denoising
dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey()
