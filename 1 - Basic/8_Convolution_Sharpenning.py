import pandas as pd
import numpy as np
import cv2


image = cv2.imread('demo_5.jpeg')
# Create sharpenning kernel

kernel_sharpenning = np.array([[-1,-1,-1],
                               [-1,9,-1],
                               [-1,-1,-1]])

# applying different kernels
sharpened = cv2.filter2D(image, -1, kernel_sharpenning)

cv2.imshow('Sharpened', sharpened)
cv2.waitKey()
