import pandas as pd
import numpy as np
import cv2

image = cv2.imread('demo_5.jpeg')
#image = cv2.imread('./images/input.jpg')

# Create a matrix of ones, then multiply it by a scaler of 100
# This gives a matrix with same dimensions of our image with all values being 100
M = np.ones(image.shape, dtype = 'uint8') * 75

# using this to add this matrix M to our image
# increase in brightness
added = cv2.add(image, M)
cv2.imshow('Added', added)
cv2.waitKey()

# Substract
# decrease in brightness
sub = cv2.subtract(image, M)
cv2.imshow('Sub', sub)
cv2.waitKey()

##################################
# BITWISE OPERATIONS AND MASKING #
##################################

# Making Square
# if colored images then np.zeros((300,300,3), np.uint8)
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50,50), (250,250), 255, -2)
cv2.imshow("Square", square)
cv2.waitKey()

# Making Ellipse
ellipse = np.zeros((300,300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150,150), 30, 0, 180, 255, -1)
cv2.imshow('Ellipse', ellipse)
cv2.waitKey()

# Bitwise Opertions (AND, OR, XOR, NOT)

# and
And = cv2.bitwise_and(square, ellipse)
cv2.imshow('AND', And)
cv2.waitKey()

# or
bitOr = cv2.bitwise_or(square, ellipse)
cv2.imshow('OR', bitOr)
cv2.waitKey()

# xor
bitXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow('XOR', bitXor)
cv2.waitKey()

# not (tudo que não é)
bitnot = cv2.bitwise_not(square)
cv2.imshow('NOT', bitnot)
cv2.waitKey()
