# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:24:05 2021

@author: wngks
"""
import cv2,math,copy,imutils
import numpy as np
from numpy import dot
from numpy.linalg import norm

def draw_cube(img,rvecs,tvecs,mtx,dist,l):
    axis = np.float32([[-l,l, 0], [l, l, 0], [l, -l, 0], [-l, -l, 0],
						   [-l,l, l], [l, l, l], [l, -l, l], [-l, -l, l]])
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), -2)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),1)    
    return img, imgpts

def get_depth(img,pt):
    x1,y1 = pt[0]
    x2,y2 = pt[2]
    xm = int((x1+x2)/2)
    ym = int((y1+y2)/2)
    return img[ym, xm]/1000

# marker detector and draw box, id
def detect_aruco_corners(IMAGE,arucoDict,arucoParms,fontFace,fontScale,color,mtx,dist,dep,s=1):
    corners,ids,rejected = cv2.aruco.detectMarkers(IMAGE,arucoDict,parameters = arucoParms)   #detect
    img_pts = []  
    cv2.aruco.drawDetectedMarkers(IMAGE, corners) 

    if len(corners):
        for _,i in enumerate(corners):
            cv2.putText(IMAGE, str('id:{}'.format(int(ids[_]))), (int(i[0][2][0]),int(i[0][2][1])), fontFace, fontScale, color)
            
    if s != 0:
        if len(corners):
            rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            for i in range(0, ids.size):
                # draw axis for the aruco markers
                cv2.aruco.drawAxis(IMAGE, mtx, dist, rvec[i], tvec[i], 0.02)
                IMAGE ,pts = draw_cube(IMAGE, rvec[i], tvec[i], mtx, dist, 0.025)
                
                depth = get_depth(dep,pts[4:])
                cv2.putText(IMAGE, str('dep:{} m'.format(float(depth))), tuple(pts[1]), fontFace, fontScale, (0,255,0))
                img_pts.append(pts)

    return IMAGE,corners,ids,img_pts

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): 
    
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = color3
    else: 
        color = color1
        
    ROI_image = cv2.fillPoly(mask, vertices, color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

#filter_red
def filter_red_yello(img,lower1,upper1,lower2,upper2,y_lower1,y_upper1):
    
    lower_mask = cv2.inRange(img, lower1, upper1)
    upper_mask = cv2.inRange(img, lower2, upper2)
    yellow_lower = cv2.inRange(img,y_lower1,y_upper1)
    
    full_mask = lower_mask+upper_mask + yellow_lower
    # plt.subplot(339),plt.imshow(full_mask),plt.title('RED')
    hsv_img = cv2.bitwise_and(img, img, mask = full_mask)
    
    img2 = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
    
    return img2

def get_second_largest(arr):
    first_largest = None
    second_largest = None 
    for e in arr:
        if first_largest is None: 
            first_largest = e
        else: 
            if e > first_largest: 
                second_largest = first_largest 
                first_largest = e 
            else: 
                if second_largest is None: 
                    second_largest = e 
                else: 
                    if e > second_largest: 
                        second_largest = e 
    return second_largest

def make_lane_block(img,edge_img,box_position,interval_x,num):
    # edge_img histogram_interval
    interval_x = [110,180,250,320,390,460,530,600]
    interval_y = [180,135,90,45]
    # Histogram for each pixel section in the image
    hist = []
    for i in interval_x[1:]:
        f = edge_img[150:,i-70:i]
        hist.append(f.sum()/100)
    ms = hist.index(max(hist))
    mss = hist.index(get_second_largest(hist))

def cos_sim(A, B):   
    return dot(A, B)/(norm(A)*norm(B))

def line_d(line): 
    x1,y1,x2,y2 = line
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def inv_roi(img, vertices, color3=(0,0,0), color1=0):   
    if len(img.shape) > 2: # if color channel :
        color = color3
    else: # if gray scale :
        color = color1

    ROI_image = cv2.fillPoly(img, vertices, color)

    return ROI_image

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def put_picture(image,frame,refpts):
    #image : original image
    # frame : want to input image
    #refpts : Area to put in the original image
    h,w = image.shape[:2]

    sh,sw = frame.shape[:2]
    srcMat = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    (L,R,BR,BL) = refpts[0]
    dstMat = [L, R, BR, BL]
    dstMat = np.array(dstMat)

    (H, _) = cv2.findHomography(srcMat, dstMat)
    
    warped = cv2.warpPerspective(frame, H, (w, h))
    mask = np.zeros((h, w), dtype="uint8")
    
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),cv2.LINE_AA)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    
    imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
    
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    return output


def copy_img(img):
    return copy.deepcopy(img)

def dist(P,graph):
    
    d = abs(graph[0]*P[0] + graph[1]*P[1] + graph[2])/(graph[0]**2+graph[1]**2)**0.5
    return d

def plot_yolo_info(img, results, fontFace, fontScale):
    result_pd = results.pandas().xyxy[0]  # Pandas DataFrame
    result_np = result_pd.to_numpy()
    for i in range(len(result_np)):
        pt = np.array([round(result_np[i][0]),round(result_np[i][1]),round(result_np[i][2]),round(result_np[i][3])])
        color = [255,0,0]
        img = cv2.rectangle(img, (pt[0],pt[1]),(pt[2],pt[3]),color,thickness = 2)
        cv2.putText(img, str('{}'.format(result_np[i][6])),(pt[0]+40,pt[1]), fontFace, fontScale, (0,255,0))
    return img

# ar player 
def ar_player(image,frame,refpts,i):
    h,w = image.shape[:2]
    frame = imutils.resize(frame,width=300)
    frame = np.array(frame)
    sh,sw = frame.shape[:2]
    srcMat = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]]) 
    
    (L,R,BR,BL) = refpts[i]
    dstMat = [L, R, BR, BL] 
    dstMat = np.array(dstMat) 
    (H, _) = cv2.findHomography(srcMat, dstMat) 
    warped = cv2.warpPerspective(frame, H, (w, h))
    mask = np.zeros((h, w), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),cv2.LINE_AA)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)

    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    # output = cv2.warpPerspective(output, minv, (w, h))
    return output



#if __name__=="main":
#    print("Vision project module")
