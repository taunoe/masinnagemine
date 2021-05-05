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

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# GRayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Num of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

# Mask
blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
mask = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow('Mask', mask)

# Masked image grayscale histogram
mask_hist = cv.calcHist([gray], [0], mask, [256], [0,256])
plt.figure()
plt.title('Mask Histogram')
plt.xlabel('Bins')
plt.ylabel('Num of pixels')
plt.plot(mask_hist)
plt.xlim([0, 256])
plt.show()

# Colour image
src = cv.imread(img_file)

# Colour histogram
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Num of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([src], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()

# Mask
blank = np.zeros(src.shape[:2], dtype='uint8')
circle = cv.circle(blank, (src.shape[1]//2, src.shape[0]//2), 100, 255, -1)
mask = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow('Mask', mask)

# Mask Colour histogram
plt.figure()
plt.title('Mask Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Num of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([src], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()

####################################################################
if __name__ == "__main__":
    print("Tere!")


    cv.waitKey(0)