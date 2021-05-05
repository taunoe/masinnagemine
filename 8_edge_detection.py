#!/usr/bin/env python3
'''
Tauno Erik
05.05.2021
pip install opencv-contrib-python   # 
pip install caer                    # https://github.com/jasmcaus/caer
pip install matplotlib

Sources:https://www.youtube.com/watch?v=oXlwWbU8l2o
        https://github.com/jasmcaus/opencv-course

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
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Edge Detection
# 1. Laplacion
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacion', lap)

# 2. Sobel
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)
sobel_combined = cv.bitwise_or(sobel_x, sobel_y)
cv.imshow('Sobel x', sobel_x)
cv.imshow('Sobel y', sobel_y)
cv.imshow('Sobel combined', sobel_combined)

# 3. Canny
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)


####################################################################
if __name__ == "__main__":
    print("Tere!")


    cv.waitKey(0)