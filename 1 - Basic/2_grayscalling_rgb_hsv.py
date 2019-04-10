import pandas as pd
import numpy as np
import os
import cv2

image = cv2.imread('./images/input.jpg')
image = cv2.imread('teste.png')

# verify if image is RGB
B, G, R = image[0,0]
print(B, G, R)


# we use cvt color to convert
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray Scale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

# another better method is load the image into a gray scale
gray_image = cv2.imread('./images/input.jpg',0)

cv2.imshow('Gray Scale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()

# after grayscalling (2 dimensional)
print(gray_image.shape)

# RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv_image)
cv2.waitKey()
cv2.imshow('HSV', hsv_image[:,:,0])
cv2.waitKey()
cv2.imshow('HSV', hsv_image[:,:,1])
cv2.waitKey()
cv2.imshow('HSV', hsv_image[:,:,2])
cv2.waitKey()
cv2.destroyAllWindows()


# visualizing each channel of RGB
# remove all color density
B, G, R = cv2.split(image)

zeros = np.zeros(image.shape[:2], dtype = 'uint8')

cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.waitKey()
cv2.imshow("Red", cv2.merge([zeros, G, zeros]))
cv2.waitKey()
cv2.imshow("Red", cv2.merge([B, zeros, zeros]))
cv2.waitKey()
