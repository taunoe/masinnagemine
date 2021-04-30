#!/usr/bin/env python3
'''
Tauno Erik
30.04.2021
pip install opencv-contrib-python   # 
pip install caer                    # https://github.com/jasmcaus/caer

Sources:https://www.youtube.com/watch?v=oXlwWbU8l2o
        https://github.com/jasmcaus/opencv-course

Deep learning:https://www.youtube.com/watch?v=VyWAvY2CF9c
'''

import os
import cv2 as cv

def full_path(filename):
    ''' Returns full path to file. '''
    folder = os.path.dirname(__file__) # File location
    full_path = os.path.join(folder, filename)
    return full_path

def read_show_img():
    ''' Read and show example image. '''
    img_file = full_path('images/cat.jpg')
    img = cv.imread(img_file)
    cv.imshow('Cat', img)
    cv.waitKey(0)

def read_show_video():
    ''' Read and show example video '''
    video_file = full_path('images/cat.mp4')
    capture = cv.VideoCapture(video_file) # 0, 1, 2 if webcam

    while True:
        isTrue, frame = capture.read()
        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF==ord('q'): # Press q to quit
            break

    capture.release()
    cv.destroyAllWindows()

####################################################################
if __name__ == "__main__":
    #read_show_img()
    read_show_video()
