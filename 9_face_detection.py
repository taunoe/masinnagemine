#!/usr/bin/env python3
'''
Tauno Erik
05.05.2021
pip install opencv-contrib-python   # 
pip install caer                    # https://github.com/jasmcaus/caer
pip install matplotlib

Sources:https://www.youtube.com/watch?v=oXlwWbU8l2o
        https://github.com/jasmcaus/opencv-course
        https://github.com/opencv/opencv/tree/master/data/haarcascades
        https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

'''

import cv2 as cv
#import numpy as np
#import matplotlib.pyplot as plt

# Face Detection
# Haarcascade
haar_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

img1 = cv.imread('images/lennu1.jpg')
img2 = cv.imread('images/grupp.jpg')
#cv.imshow('Lennart Meri', img1)
#cv.imshow('Lennart Meri', img2)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#cv.imshow('Lennart Meri 1', gray1)
#cv.imshow('Lennart Meri 2', gray2)

# Leia näod
faces_rect1 = haar_cascade.detectMultiScale(gray1, scaleFactor=1.2, minNeighbors=3)
print(f'gray1 - Leitud nägude arv: {len(faces_rect1)}')

faces_rect2 = haar_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=3)
print(f'gray2 - Leitud nägude arv: {len(faces_rect2)}')

# joonista ristkülik
for (x,y,w,h) in faces_rect1:
    cv.rectangle(img1, (x,y), (x+w,y+h,), (0,255,0), thickness=2)

for (x,y,w,h) in faces_rect2:
    cv.rectangle(img2, (x,y), (x+w,y+h,), (0,255,0), thickness=2)

# Kuva
cv.imshow('Leitud näod 1', img1)
cv.imshow('Leitud näod 2', img2)


cv.waitKey(0)
####################################################################
if __name__ == "__main__":
    print("Tere!")
