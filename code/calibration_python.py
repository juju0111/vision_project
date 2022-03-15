"""
Created on Thu Jun 10 00:08:57 2021

@author: wngks
"""

import numpy as np
import cv2,os

dir = os.getcwd()

os.chdir('/home/park/vision_project/calibration')
file_dir = '/home/park/vision_project/calibration'
file_names = os.listdir(file_dir)

file_names.sort()

print(file_names)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# images = glob.glob(file_dir)
for fname in file_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(8,6),(-1,-1),criteria)
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        h,  w = img.shape[:2]
        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow('img',img)
    cv2.imshow('dst',dst)
    
    cv2.waitKey(200)

cv2.waitKey(0)
cv2.destroyAllWindows()