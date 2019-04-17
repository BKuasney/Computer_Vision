import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('input.jpg')
image = cv2.imread('tobago.jpg')

# calcHist(images, channels, mask, histSize, ranges)s
histogram = cv2.calcHist([image],[0], None, [256], [0,256])

# we plot a histogram, ravel() flatens our image array
plt.hist(image.ravel(), 256, [0,256]); plt.show()

# View separete color channels
color = ('b', 'g', 'r')

# We now separate the colors and plot each in the histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])

# distributtion of colors
plt.show()
