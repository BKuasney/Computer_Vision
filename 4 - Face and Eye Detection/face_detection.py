import pandas as pd
import numpy as np
import cv2

# load HAAS cascade
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image and then convert into a grayscale
image = cv2.imread('Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordinates
faces = face_classifier.detectMultiScale(gray, 1.3, 5)
# print(faces) to see the location of face

# When no faces detected, face_classifier returns an empty tuple
if faces is ():
    print('No faces Found')

# We iterate through our faces array and draw a rectangle
# over each faces in faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (127, 0, 255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
    Combine face and eye detection
'''

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# when no faces
if faces is ():
    print('No faces')

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (127, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0))
        cv2.imshow('img', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()


'''
    Live Face and Eye Detection
'''

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x,y,w,h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    roi_color = cv2.flip(roi_color,1)
    return roi_color

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: # enter key
        break

cap.release()
cv2.destroyAllWindows()
