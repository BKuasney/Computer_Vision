import cv2
import numpy as np
import pandas as pd
import pathlib

# load image using imread
input = cv2.imread('input.jpg')

# Load an image on window and wait to close
# hello world is the title window
cv2.imshow('hello world',input)
cv2.waitKey()
cv2.destroyAllWindows()

print(input.shape)
print('Height of Image {} pixels'.format(int(input.shape[0])))
print('Height of Image {} pixels'.format(int(input.shape[1])))

# to save a image
cv2.imwrite('./Output/output.jpg', input)
