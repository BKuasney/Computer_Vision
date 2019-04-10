import cv2
import numpy as np

img = cv2.imread("teste.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img[img >= 128]= 255
img[img <40] = 255
cv2.imwrite('out.jpg', img)

im = cv2.imread('teste.png')
im[np.where((im == [38,38,38]).all(axis = 2))] = [255,255,255]
cv2.imwrite('output.png', im)
