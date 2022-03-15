"""
Created on Sun Jun 13 18:39:59 2021

@author: wngks
"""
import math,os
os.chdir("/home/park/vision_project/code")
import numpy as np
import torch
import cv2,sys,copy,imutils,itertools
from numpy import dot
from numpy.linalg import norm
from vision_project_module import *
import matplotlib.pyplot as plt

os.chdir('/home/park/vision_project/RAW')
file_dir = '/home/park/vision_project/RAW'
file_names = os.listdir(file_dir)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

raw_imgs = []
depth = []

dica = cv2.VideoCapture('dica.gif')
chic1 = cv2.VideoCapture('chicken.gif')
chic2 = cv2.VideoCapture('chicken.gif')

# cv2.Rodrigues(rvec)
for i in file_names:
    if 'depth' in i :
        depth.append(i)
    if 'image' in  i:
        raw_imgs.append(i)


depth.sort()
raw_imgs.sort()

#camera instrinsic matrix and distortion coefficient 
mtx = np.array([[601.73815588,   0.        , 300.35016164],[  0.        , 610.19885083, 236.579869  ],[0,0,1]])
dist = np.array([[ 0.09222483,  0.00764416,  0.00203016, -0.00808426, -0.50662685]])

img = cv2.imread(raw_imgs[0])
h,w = img.shape[:2]
h = h-300
# Points for direct homography of bird eye view
src = np.float32([[0, h], [w, h], [w, 0], [0, 0]])
dst = np.float32([[202, h], [w-185, h], [w, 0], [0, 0]])

# various for aruco 
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParms = cv2.aruco.DetectorParameters_create()
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
clr = (255,0,0)

r_lower1 = np.array([0, 50, 100])
r_upper1 = np.array([4, 255, 255])
r_lower2 = np.array([160,50,100])
r_upper2 = np.array([180,255,255])
#yellow 
y_lower1 = np.array([15, 90, 130])
y_upper1 = np.array([24, 180, 255]) 

#red filter array # upper boundary RED color range values; Hue (160 - 180)
max_degree = 90

#Id detected on the floor
floor_id=[]


# Variables to save for the panorama
img_for_panorama = []
save_c =0


#Image taken to show the direction of the arrow
arrow = cv2.imread('arrow.jpg')
arrow = cv2.resize(arrow,(100,100))
arrow  =cv2.rotate(arrow, cv2.ROTATE_180)

ah,aw = arrow.shape[:2]
ah2 = int(ah/2)
aw2 = int(aw/2)

for count,i in enumerate(raw_imgs):
    img = cv2.imread(i)
    results = model(img)  # inference
    
    # The copy to draw the image
    img2 = copy_img(img)
    img3 = copy_img(img)
    
    #depth image
    dep = cv2.imread(depth[count],-1)
    
    # Starting from roi 300, I want to see only the crop line from the bottom.
    roi = img[300:,:]
    blank_img = np.zeros(roi.shape)
    
    #roi shape
    h,w = roi.shape[:2]
    
    # trapezoidal polygon to crop
    vertices = np.array([[(100,0),(w-100,0),(w,h),(0,h)]],dtype = np.int32)
    
    # crop 
    roi = region_of_interest(roi, vertices)
    
    # Use bird eye view homography to make it look straight.
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    # inverse homography
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
    # warping
    warped_img = cv2.warpPerspective(roi, M, (w, h)) # Image warping
    # Copy of warping image - Lines are not drawn, but gifs will be included in the qr code
    warped_img2 = copy_img(warped_img)
   
    hsv = cv2.cvtColor(warped_img,cv2.COLOR_BGR2HSV)
    filter_lane = filter_red_yello(hsv,r_lower1,r_upper1,r_lower2,r_upper2,y_lower1,y_upper1)


    #threshold 
    __,thresh = cv2.threshold(filter_lane,70,255,cv2.THRESH_BINARY)

    blur = cv2.GaussianBlur(thresh,(3,3),5)
    edge_img2 = cv2.Canny(blur,50,100)
    # get hough line 
    lines = cv2.HoughLinesP(edge_img2,1,np.pi/180,40,minLineLength=5,maxLineGap=20)
    
    line_arr = np.squeeze(lines)
    
    
    line_dict = {}
    if lines is not None:
        l = len(lines) 
        for i in range(l):
            if l == 1:
                line_dict[i] = line_arr
            elif l > 1 :
                line_dict[i] = line_arr[i]    
        # The reason for creating a combination is to check the similarity of each extracted line so that the vector direction is similar.
        # I put this syntax to filter out lines.
        # Create a combination of lines
        combination = list(itertools.combinations(line_dict,2))
        
        filtered_line = {}  # Where to put the filtered line
        # If there is no combination ? -> Only one line came out.
        if len(combination) == 0:
            filtered_line[0] = line_dict[0]  
        # If a line with high similarity is found by comparing the combined lines
        # Only the longer lines are put in filtered_line.
        else:
            for i in combination:
                rate = cos_sim(line_dict[i[0]], line_dict[i[1]])
                if rate > 0.98:
                    l1  = line_d(line_dict[i[0]])
                    l2  = line_d(line_dict[i[1]])
                    if l1>l2:
                        filtered_line[i[0]] = line_dict[i[0]]
                    else:
                        filtered_line[i[1]] = line_dict[i[1]]
                        
        slopes = []
        # Find the angle among the filtered lines and filter out all those larger than 170 degrees.
        if filtered_line != {}:
            for c,line in enumerate(filtered_line):
                x1,y1,x2,y2 = filtered_line[line]
                # y1 += 300
                # y2 += 300
                ang = (np.arctan2(y1 - y2, x1 - x2) * 180) / np.pi
                if abs(ang)<170:
                    cv2.line(blank_img,(x1,y1),(x2,y2),(255,0,0),3,cv2.LINE_AA)
                    # cv2.line(warped_img,(x1,y1),(x2,y2),(255,0,0),3,cv2.LINE_AA)

                    slopes.append(ang)
    
        if slopes:
            a= list(np.abs(slopes))
            idx = a.index(max(a))
            max_degree = slopes[idx]
            
    else:
        max_degree = 90
    # arrow Set direction !
    if max_degree >0:    # Turn right ! 
        rotate = max_degree-90
        Arw = cv2.getRotationMatrix2D((aw2,ah2),-rotate,1)
        rotated_arrow =cv2.warpAffine(arrow,Arw,(aw,ah))    
    
    else:              # Trun lefg ! 
        rotate = max_degree + 90
        Arw = cv2.getRotationMatrix2D((aw2,ah2),-rotate,1)
        rotated_arrow =cv2.warpAffine(arrow,Arw,(aw,ah))

    # Return the image with the bottom line to the image before homography with inversewarp

    # Only lines are drawn on an empty image -> The image is merged with the original image at the end.
    blank_img = cv2.warpPerspective(blank_img, Minv, (w, h))
    blank_img = np.uint8(np.vstack((np.zeros((300,640,3)),blank_img)))

    vertices = np.array([[(100,300),(w-100,300),(w,480),(0,480)]],dtype = np.int32)
    zero = inv_roi(img3, vertices)

    # Marker detect
    #Extract floor markers from warp images
    __,corners_,Id_,__= detect_aruco_corners(warped_img2,arucoDict,arucoParms,fontFace,fontScale,clr,mtx,dist,dep,0)
    #Extract all markers from unwarped images - Play chicken gifs for those that are not the bottom of the extracted ones
    pose_img,corners,Id,img_pts= detect_aruco_corners(img2,arucoDict,arucoParms,fontFace,fontScale,clr,mtx,dist,dep)
    
    refpts= []  

    # If there is a detected box, execute the if statement
    if len(corners) > 0:
        # squeeze and make a list
        idd = np.squeeze(Id) # ID from the original image
        idd = idd.tolist()
        idd_ = np.squeeze(Id_) # # ID from bird eye view
        idd_ = idd_.tolist()
    
        ret, f1 = dica.read()
        ret2,f2 = chic1.read()

        frame = [f1,f2]

        if not ret or not ret2:
            continue
        
        if type(idd) == int:
            idd = [idd]
        if type(idd_) == int:
            idd_ = [idd_]    
        if idd_ != None:
            # Put the ID from bird_eye_view into floor_id.
            if idd_[0] not in floor_id:
                floor_id.append(idd_[0])
        output = pose_img
        for _,i in enumerate(idd):
            corner = np.squeeze(img_pts[_][4:])
            refpts.append(corner)
            # Dicaprio on the floor, chicken on the side
            if i in floor_id:
                output = ar_player(output,frame[0],refpts,_)
            else:
                output = ar_player(output,frame[1],refpts,_)

    # If no marker is detected, just pose img
    else:
        output = pose_img
    
    # save warped image for making panorama
    # print(count)
    # if count<10:
    #     cv2.imwrite('../warp/0000{}_warped.jpg'.format(count+1),warped_img)
    # else:
    #     cv2.imwrite('../warp/000{}_warped.jpg'.format(count),warped_img)
    
    
    blank_img = np.uint8(blank_img)
    total = weighted_img(blank_img,output,1,1)
    
    # Trapezoidal area to put the arrow in
    space = [np.array([(560, 380), (600, 380), (640, 480), (520, 480)])]
    # put an arrow
    total = ar_player(total,rotated_arrow,space,0)
    total = plot_yolo_info(total, results,fontFace, fontScale)

    cv2.imshow("total",total)


    if cv2.waitKey(1) ==27:
        break
    cv2.waitKey(10)
   
cv2.destroyAllWindows()


# make panorama

# os.chdir('/home/park/vision_project/warp')

# file_dir = '/home/park/vision_project/warp'

# file_names = os.listdir(file_dir)

# file_names.pop(-1)

# imgs=[]
# for c,name in enumerate(file_names):
#     if c %3 ==0:
#         continue
#     else:
#         img = cv2.imread(name)
        
#         if img is None:
#             print('Image load failed!')
#             #sys.exit()
#             continue
            
#         imgs.append(img)
    
# stitcher = cv2.Stitcher_create()

# status, dst = stitcher.stitch(imgs)


# if status != cv2.Stitcher_OK:
#     print('Stitch failed!')
#     sys.exit()
    
# cv2.imwrite('output.jpg', dst)

# cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()
