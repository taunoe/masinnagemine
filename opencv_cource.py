#!/usr/bin/env python3
'''
Tauno Erik
04.05.2021
pip install opencv-contrib-python   # 
pip install caer                    # https://github.com/jasmcaus/caer

Sources:https://www.youtube.com/watch?v=oXlwWbU8l2o
        https://github.com/jasmcaus/opencv-course

Deep learning:https://www.youtube.com/watch?v=VyWAvY2CF9c
'''

import os
import cv2 as cv
import numpy as np

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


def rescale_frame(frame, scale=0.75):
    ''' Rescale video/image/live video size. '''
    height = int(frame.shape[0]*scale) # convert float to int
    width = int(frame.shape[1]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def change_res(width, height):
    ''' Rescale live video '''
    capture.set(3, width)
    capture.set(4, height)


def rescale_show_video():
    ''' '''
    video_file = full_path('images/cat.mp4')
    capture = cv.VideoCapture(video_file) # 0, 1, 2 if webcam
    while True:
        isTrue, frame = capture.read()
        frame_resized = rescale_frame(frame, scale=0.5)
        # Two windows
        cv.imshow('Orig video', frame)
        cv.imshow('Resized video', frame_resized)

        if cv.waitKey(20) & 0xFF==ord('q'): # Press q to quit
            break
    capture.release()
    cv.destroyAllWindows()


def draw():
    # generate blank image
    blank_img = np.zeros((500, 500, 3), dtype='uint8')
    #cv.imshow('Blank', blank_img)
    
    # 1. Paint the image a certain colour
    #blank_img[:] = 0,255,0 # RGB
    #cv.imshow('Green', blank_img)
    # colour some area
    #blank_img[200:300, 300:400] = 0,0,255 # RGB
    #cv.imshow('Red rectangle', blank_img)

    # 2. Draw a Rectangle
    pt1 = (10, 10)
    pt2 = (250, 250)
    color = (0, 250, 0)
    cv.rectangle(blank_img, pt1, pt2, color, thickness=2)
    #cv.imshow('Rectangle', blank_img)

    pt1 = (250, 250)
    pt2 = (350, 350)
    color = (0, 250, 0)
    cv.rectangle(blank_img, pt1, pt2, color, thickness=cv.FILLED) # or -1
    #cv.imshow('Rectangle', blank_img)

    # 3. Draw a circle
    center = (blank_img.shape[1]//2, blank_img.shape[0]//2)
    radius = 40
    color = (0, 0, 255)
    cv.circle(blank_img, center, radius, color, thickness=3)
    #cv.imshow('Circle', blank_img)

    # 4. Draw a line
    pt1 = (50, 50)
    pt2 = (450, 100)
    color = (0, 200, 200)
    cv.line(blank_img, pt1, pt2, color, thickness=3)
    #cv.imshow('Line', blank_img)

    # Write text
    fontFace = cv.FONT_HERSHEY_TRIPLEX
    fontScale = 1.0
    color = (255, 255, 255)
    cv.putText(blank_img, 'Tere maailm!', (100, 100), fontFace, fontScale, color, thickness=2)
    cv.imshow('Tekst',blank_img)

    cv.waitKey(0)


def convert_grayscale():
    img = cv.imread(full_path('images/cat.jpg'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale", gray)
    cv.waitKey(0)

def blur_img():
    img = cv.imread(full_path('images/cat.jpg'))
    # Gaussian blur
    ksize = (9, 9) # odd number
    blur = cv.GaussianBlur(img, ksize, cv.BORDER_DEFAULT)
    cv.imshow("Gaussian blur", blur)
    cv.waitKey(0)

def edge_cascade():
    img = cv.imread(full_path('images/cat.jpg'))
    canny = cv.Canny(img, 125, 175)
    cv.imshow("Canny Edge", canny)
    cv.waitKey(0)

    # et servi oleks siis blurida enne
    blur_canny = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
    canny2 = cv.Canny(blur_canny, 125, 175)
    cv.imshow("Canny Edge + blur", canny2)
    cv.waitKey(0)

def dilating_img():
    img = cv.imread(full_path('images/cat.jpg'))
    blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)
    dilated = cv.dilate(canny, (3, 3), iterations=3)
    cv.imshow('Dilated', dilated)
    cv.waitKey(0)

def resize_img():
    img = cv.imread(full_path('images/cat.jpg'))
    resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA) # Kui suurendada INTER_LINEAR või INTER_CUBIC
    cv.imshow('Resized', resized)
    cv.waitKey(0)

def cropping_img():
    img = cv.imread(full_path('images/cat.jpg'))
    cropped = img[50:200, 200:400]
    cv.imshow('Cropped', cropped)
    cv.waitKey(0)

def translate(img, x, y):
    '''
    -x --> Left
     x --> Right
    -y --> Up
     y --> Down
    '''
    trans_matrix = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0]) # width, height
    return cv.warpAffine(img, trans_matrix, dimensions)

def translation():
    img = cv.imread(full_path('images/cat.jpg'))
    translated = translate(img, 100, 100)
    cv.imshow('Translated', translated)
    cv.waitKey(0)

def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    if rot_point is None:
        # center point
        rot_point = (width//2, height//2)

    rot_matrix = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dimensions = (width, height) # width, height

    return cv.warpAffine(img, rot_matrix, dimensions)

def rotating_img():
    img = cv.imread(full_path('images/cat.jpg'))
    rotated = rotate(img, 35)
    cv.imshow('Rotated', rotated)
    cv.waitKey(0)

def flipping():
    img = cv.imread(full_path('images/cat.jpg'))
    vert = 0
    hort = 1
    vert_hort = -1
    fliped = cv.flip(img, vert_hort)
    cv.imshow('Flipped', fliped)
    cv.waitKey(0)


def contours1():
    img = cv.imread(full_path('images/cat.jpg'))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)

    #contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)         # hirarical
    #contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)     # external contours
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)          # all contours
    # cv.CHAIN_APPROX_NONE

    print(f'{len(contours)} contours found')

    cv.imshow('Contours', canny)
    cv.waitKey(0)

def contours2():
    img = cv.imread(full_path('images/cat.jpg'))
    blank = np.zeros(img.shape, dtype='uint8')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

    #contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)         # hirarical
    #contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)     # external contours
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)          # all contours
    # cv.CHAIN_APPROX_NONE

    print(f'{len(contours)} contours found')
    cv.imshow('Thresh', thresh)

    cv.drawContours(blank, contours, -1, (0,255,0), 1)
    cv.imshow('Kontuurid', blank)

    cv.waitKey(0)
   
####################################################################
if __name__ == "__main__":
    print("Tere!")
    #read_show_img()
    #read_show_video()
    #rescale_show_video()
    #draw()
    #convert_grayscale()
    #blur_img()
    #edge_cascade()
    #dilating_img()
    #resize_img()
    #cropping_img()
    #translation()
    #rotating_img()
    #flipping()
    #contours1()
    contours2()
