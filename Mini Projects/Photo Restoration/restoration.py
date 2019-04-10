import pandas as pd
import numpy as np
import cv2

# load damaged photo
image = cv2.imread('abraham.jpg')
cv2.imshow('Original', image)
cv2.waitKey(0)

# Load photo where we have marked the damaged areas
marked_damages = cv2.imread('mask.jpg', 0)
cv2.imshow('Marked Damages', marked_damages)
cv2.waitKey(0)

# make a mask out of our marked image be changing all colors
# that are not white, to black
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

# Dilate our the marks we made
# since thresholding has narrowed it slightly
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations=1)
cv2.imshow('Dilated Mask', mask)
cv2.imwrite('abaham_mask.png', mask)
cv2.waitKey(0)

restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()
