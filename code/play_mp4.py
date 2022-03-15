# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:33:54 2021

@author: wngks
"""
import numpy as np
import cv2,os,imutils,copy
from  PIL import Image
os.chdir('/home/park/vision_project/RAW')

capture = cv2.VideoCapture('dica.gif')

while capture.isOpened():
    run, frame = capture.read()
    frame = np.array(frame)

    if not run:
        print("[Frame cannot be received] - Ends")
        break

    cv2.imshow('video',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
 
capture.release()
cv2.destroyAllWindows()