#!/usr/bin/env python3
'''
Tauno Erik
30.04.2021
Requirments    pip install opencv-contrib-python

Insp: https://www.youtube.com/watch?v=UOIKXp82iEw
'''

import cv2
import winsound
import os

def returnCameraIndexes():
    # https://stackoverflow.com/questions/8044539/listing-available-devices-in-python-opencv
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def display_diff():
    '''
    Displays only difference between frames.
    '''
    try:
        while cam.isOpened():
            ret, frame1 = cam.read()
            ret, frame2 = cam.read()
            diff = cv2.absdiff(frame1, frame2)
            # To exit press 'q'
            if cv2.waitKey(10) == ord('q'):
                break
            # Display
            cv2.imshow('Erinevus', diff)
    except:
        print("Error.")


def display_contoures():
    '''
    Displays countoures around moving parts.
    '''
    try:
        while cam.isOpened():
            # Read frames (images) from camera
            ret, frame1 = cam.read()
            ret, frame2 = cam.read()
            # Compare frames
            diff = cv2.absdiff(frame1, frame2)
            # Convert diff to grayscale image
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            # Blur gray image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Converts to Binary images. Only black and white colour.
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # Expand moving image part
            dilated = cv2.dilate(thresh, None, iterations=3)
            # Find moving part contures
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Display contours
            cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) #

            # To exit press 'q'
            if cv2.waitKey(10) == ord('q'):
                break
    
            # Display
            cv2.imshow('Kaamera', frame1)
    except:
        print("Error.")


def detect_movement():
    '''
    Draws recangles around moving parts
    and plays sound when something will move.
    '''
    try:
        while cam.isOpened():
            # Read frames (images) from camera
            ret, frame1 = cam.read()
            ret, frame2 = cam.read()
            # Compare frames
            diff = cv2.absdiff(frame1, frame2)
            # Convert diff to grayscale image
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            # Blur gray image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Converts to Binary images. Only black and white colour.
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # Expand moving image part
            dilated = cv2.dilate(thresh, None, iterations=3)
            # Find moving part contures
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                # Select movement size area.
                # If contour is smaller it will be ignored.
                if cv2.contourArea(c) < 2000:
                    continue
                # Contour position and size
                x, y, w, h = cv2.boundingRect(c)
                # Draw rectangle
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Play sound
                winsound.PlaySound(soundfile, winsound.SND_ASYNC)
                #winsound.PlaySound('alert.wav', winsound.SND_FILENAME|winsound.SND_ASYNC)
                #winsound.PlaySound('SystemAsterisk', winsound.SND_ALIAS|winsound.SND_ASYNC)
                #winsound.Beep(500, 100) # it will bloks video 

            # To exit press 'q'
            if cv2.waitKey(10) == ord('q'):
                break
    
            # Display
            cv2.imshow('Kaamera', frame1)
    except:
        print("Error.")


if __name__ == "__main__":
    print('To exit press "q"')

    dir = os.path.dirname(__file__) # File location
    
    soundfile = os.path.join(dir, 'alert.wav')

    # Select camera. Usualy 0, or 1 and so on
    #print(returnCameraIndexes())
    cam = cv2.VideoCapture(0)

    # 1
    #display_diff()
    # 2
    #display_contoures()
    # 3
    detect_movement()
    