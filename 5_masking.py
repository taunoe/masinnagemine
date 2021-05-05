#!/usr/bin/env python3
'''
Tauno Erik
05.05.2021
pip install opencv-contrib-python   # 
pip install caer                    # https://github.com/jasmcaus/caer
pip install matplotlib

Sources:https://www.youtube.com/watch?v=oXlwWbU8l2o
        https://github.com/jasmcaus/opencv-course

Deep learning:https://www.youtube.com/watch?v=VyWAvY2CF9c
'''

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def full_path(filename):
    ''' Returns full path to file. '''
    folder = os.path.dirname(__file__) # File location
    full_path = os.path.join(folder, filename)
    return full_path

img_file = full_path('images/cat.jpg')
video_file = full_path('images/cat.mp4')

img = cv.imread(img_file)
blank = np.zeros(img.shape[:2], dtype='uint8')

mask_rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, thickness=-1)
mask_circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 200, 255, -1)

cv.imshow('Ruut', mask_rectangle)
cv.imshow('Mask: Ring', mask_circle)

masked = cv.bitwise_and(img, img, mask=mask_circle)
cv.imshow('Masked', masked)

weird_shape = cv.bitwise_and(mask_circle, mask_rectangle)
cv.imshow('weird_shape', weird_shape)

masked_weird = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked weird', masked_weird)

   
####################################################################
if __name__ == "__main__":
    print("Tere!")


    cv.waitKey(0)