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

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, thickness=-1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Ruut', rectangle)
cv.imshow('Ring', circle)

def bitwise_AND():
    ''' Intersecting regions '''
    bitwise_and = cv.bitwise_and(rectangle, circle)
    cv.imshow('Bitwise AND', bitwise_and)

def bitwise_OR():
    ''' Non-intersecting and intersecting regions '''
    bitwise_or = cv.bitwise_or(rectangle, circle)
    cv.imshow('Bitwise OR', bitwise_or)

def bitwise_XOR():
    ''' Non-intersecting regions '''
    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow('Bitwise XOR', bitwise_xor)

def bitwise_NOT():
    ''' inverts '''
    bitwise_not = cv.bitwise_not(rectangle)
    cv.imshow('Bitwise NOT', bitwise_not)

   
####################################################################
if __name__ == "__main__":
    print("Tere!")
    bitwise_AND()
    bitwise_OR()
    bitwise_XOR()
    bitwise_NOT()

    cv.waitKey(0)