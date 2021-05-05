#!/usr/bin/env python3
'''
Tauno Erik
05.05.2021
Requirments    pip install opencv-contrib-python

Insp: https://www.youtube.com/watch?v=UOIKXp82iEw
'''

import cv2 as cv
import winsound
import os

# Face Detection
# Haarcascade
haar_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def returnCameraIndexes():
    # https://stackoverflow.com/questions/8044539/listing-available-devices-in-python-opencv
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def detect_face():
    '''
    Draws recangles around moving parts
    and plays sound when something will move.
    '''
    try:
        while cam.isOpened():
            ret, img = cam.read()
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            faces_rect1 = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
          
            # joonista ristkülik
            for (x,y,w,h) in faces_rect1:
                cv.rectangle(img, (x,y), (x+w,y+h,), (0,255,0), thickness=2)

            # To exit press 'q'
            if cv.waitKey(10) == ord('q'):
                break
    
            # Display
            cv.imshow('Kaamera', img)
    except:
        print("Error.")


if __name__ == "__main__":
    print('To exit press "q"')

    dir = os.path.dirname(__file__) # File location
    
    soundfile = os.path.join(dir, 'alert.wav')

    # Select camera. Usualy 0, or 1 and so on
    #print(returnCameraIndexes())
    cam = cv.VideoCapture(0)

    detect_face()
    