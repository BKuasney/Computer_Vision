import pandas as pd
import numpy as np
import cv2

# Scketch generation function
def sketch(image)
    # convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # clean image and remove noise
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # Do an invert binarize the image
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRES_BINARY_INV)
    return mask

# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Live Sketch', sketch(frame))
    if cv2.waitKey(1) == 13 # 13 is the enter key
        break

# release camera and close windows
cap.release()
cv2.destroyAllWindows()
