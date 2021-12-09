import os
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('/Users/apple/Documents/programming/images/haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']



#features = np.load(r'/Users/apple/Documents/programming/machine learning/opencv/face recognintion/features.npy')
#labels = np.load(r'/Users/apple/Documents/programming/machine learning/opencv/face recognintion/labels.npy')


face_recognizers = cv.face.LBPHFaceRecognizer_create()
face_recognizers.read(r'/Users/apple/Documents/programming/machine learning/opencv/face recognintion/face_trained.yml')

img = cv.imread(r'/Users/apple/Documents/programming/images/Faces/val/madonna/3.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('person',gray)
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]
    label, confidence =face_recognizers.predict(faces_roi)
    print(label,confidence)

    cv.putText(img, str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.1,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness= 2)

cv.imshow('detected', img)

cv.waitKey(0)