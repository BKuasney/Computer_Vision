import pandas as pd
import numpy as np
import cv2

'''
HOUGH LINES
'''

image = cv2.imread('soduku.jpg')
cv2.imshow('Identify Shapes', image)
cv2.waitKey()

# GrayScale and Canny Edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

cv2.imshow('Canny', edges)
cv2.waitKey()

# Run HoughLines using a rho accuracy of 1 pixel
# theta accuracy of np.pi / 180 which is a 1 degree
# Our line threshold is set to 240 (number of points on line)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 140) # last parameter is qtd od lines on output

# We interate through each line and convert it to the format
# required by cv2.lines (i.e requiring end points)

for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('Hough Lines', image)
cv2.waitKey()


"""
PROBABILISTIC HOUGH LINES
"""

image = cv2.imread('soduku.jpg')
cv2.imshow('Identify Shapes', image)
cv2.waitKey()

# GrayScale and Canny Edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

cv2.imshow('Canny', edges)
cv2.waitKey()

# use the same rho and theta accuracies
# however, we specific a minimum vote (pts along line) of 100
# and Min Line lenght of 5 pixels and max gap between lines of 10 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 200, 10)
print(lines.shape)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines', image)
cv2.waitKey(0)
