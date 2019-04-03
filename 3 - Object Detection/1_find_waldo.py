import pandas as pd
import numpy as np
import cv2

# Load input image and convert to grayscale
image = cv2.imread('WaldoBeach.jpg')
cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Template image
template = cv2.imread('waldo.jpg',0)
cv2.imshow('Template', template)
cv2.waitKey()

# There are a variety of methods to perform template matching
# in this case we are using the correlation coefficient which is specifiedby the flag cv2.TM_CCOEFF
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Essentially, this function takes a “sliding window” of our waldo query image and slides it across our puzzle image
# from left to right and top to bottom, one pixel at a time.
# Then, for each of these locations, we compute the correlation coefficient to determine how “good” or “bad” the match is. 


#Create Bounding Box
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
