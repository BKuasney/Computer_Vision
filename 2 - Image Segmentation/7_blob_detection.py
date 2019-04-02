# grupo de pixels que possuem a mesma característica/propriedade
# basicamente "busca" os padrões que se repetem na imagem dado o que quer buscar

import pandas as pd
import numpy as np
import cv2

image = cv2.imread('Sunflowers.jpg', cv2.IMREAD_GRAYSCALE)

# setup the detector with default parameters
detector = cv2.SimpleBlobDetector()

# Detect blobs
keypoints= detector.detect(image)

# Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0),
                                            cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imshow('Blobs', blobs)
cv2.waitKey()
