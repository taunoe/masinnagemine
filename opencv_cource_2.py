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

# BGR is default

def convert_grayscale():
    src = cv.imread(full_path('images/cat.jpg'))
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale', gray)
    cv.waitKey(0)

def convert_hsv():
    src = cv.imread(full_path('images/cat.jpg'))
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    cv.imshow('HSV', hsv)
    cv.waitKey(0)

def convert_lab():
    src = cv.imread(full_path('images/cat.jpg'))
    lab = cv.cvtColor(src, cv.COLOR_BGR2LAB)
    cv.imshow('LAB', lab)
    cv.waitKey(0)

def plot_img():
    img = cv.imread(full_path('images/cat.jpg'))
    plt.imshow(img)
    plt.show()

def BGR_to_RGB():
    src = cv.imread(full_path('images/cat.jpg'))
    rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    cv.imshow('RGB', rgb)

    plt.imshow(rgb)
    plt.show()

    cv.waitKey(0)

def hsv_to_bgr():
    src = cv.imread(full_path('images/cat.jpg'))
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv)

    hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('hsv_bgr', hsv_bgr)

    cv.waitKey(0)

def color_channels():
    src = cv.imread(full_path('images/cat.jpg'))
    b,g,r, = cv.split(src)
    cv.imshow('Blue', b)
    cv.imshow('Green', g)
    cv.imshow('Red', r)
    print(src.shape)
    print(b.shape)
    print(g.shape)
    print(r.shape)

    merged1 = cv.merge([g,r,b])
    cv.imshow('GRB', merged1)

    merged2 = cv.merge([r,g,b])
    cv.imshow('RGB', merged2)

    merged3 = cv.merge([r,b,g])
    cv.imshow('RBG', merged3)

    merged4 = cv.merge([b,g,r])
    cv.imshow('BGR', merged4)

    cv.waitKey(0)

def color_channels_2():
    src = cv.imread(full_path('images/cat.jpg'))
    b,g,r, = cv.split(src)
    blank = np.zeros(src.shape[:2], dtype='uint8')

    blue = cv.merge([b, blank, blank])
    green = cv.merge([blank, g, blank])
    red = cv.merge([blank, blank, r])

    cv.imshow('Blue', blue)
    cv.imshow('Green', green)
    cv.imshow('Red', red)

    cv.waitKey(0)
   
####################################################################
if __name__ == "__main__":
    print("Tere!")
    #convert_grayscale()
    #convert_hsv()
    #convert_lab()
    #plot_img()
    #BGR_to_RGB()
    #hsv_to_bgr()
    #color_channels()
    color_channels_2()
