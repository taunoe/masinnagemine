#!/usr/bin/env python3
'''
Tauno Erik
04.05.2021
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


# Averaging
def average_blur():
    src = cv.imread(img_file)
    average = cv.blur(src, (7,7))
    cv.imshow('average', average)
    #cv.waitKey(0)


def gaussian_blur():
    src = cv.imread(img_file)
    gauss = cv.GaussianBlur(src, (7,7), 0)
    cv.imshow('gauss', gauss)
    #cv.waitKey(0)

def median_blur():
    src = cv.imread(img_file)
    median = cv.medianBlur(src, 7)
    cv.imshow('median', median)
    #cv.waitKey(0)

def bilateral_blur():
    src = cv.imread(img_file)
    bi = cv.bilateralFilter(src, 10, 15, 15)
    cv.imshow('bilateral', bi)
    #cv.waitKey(0)
   
####################################################################
if __name__ == "__main__":
    print("Tere!")
    #average_blur()
    #gaussian_blur()
    #median_blur()
    bilateral_blur()

    cv.waitKey(0)