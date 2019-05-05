import numpy as np
import cv2

image = cv2.imread('elephant.jpg')
image = cv2.imread('demo_5.jpeg')
cv2.imshow('Original', image)
cv2.waitKey(0)

# cv2.fastNlMeansDenoisingColored(input, None, h, hForColorComponents, templateWindowSize, searchWindowSize)
# None are - the filter strength 'h' (5-12 is a good range)
# Next is hForColorComponents, set as same value as h again
# templateWindowSize (odd numbers only) rec. 7
# searchWindowSize (odd numbers only) rec. 21



dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)

cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()
