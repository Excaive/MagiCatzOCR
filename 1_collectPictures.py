# -*- coding: utf-8 -*
# 对原始照片进行处理，通过猫的位置确定屏幕，进行透视变换，然后截取物量和FC/AC标志

import numpy as np
import cv2 as cv

#**************** 特征匹配，确定该图是魔剂猫的结算图 ****************#

picNum = 15
for i in list(range(101, 101+picNum)) + list(range(201, 201+picNum)) + list(range(301, 301+picNum)):
    stri = str(i)
    MIN_MATCH_COUNT = 150
    img1 = cv.imread('_image\\catSE2.jpg',0)              # queryImage
    img2c = cv.imread('_photos\\%s.jpg' % stri,1)        # trainImage
    img2 = cv.cvtColor(img2c, cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

#**************** 透视变换，截取屏幕 ****************#

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        M1 = cv.getPerspectiveTransform(dst, pts)

#**************** 截取物量和FC/AC标志 ****************#

        img10 = cv.warpPerspective(img2, M1, (pts[2][0][0], pts[2][0][1]))
        img10c = cv.warpPerspective(img2c, M1, (pts[2][0][0], pts[2][0][1]))

        imgAmount = img10[195:215, 255:295]
        imgAmount = cv.adaptiveThreshold(imgAmount, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 55, 12)

        imgFCAC = img10[336:356, 78:258]
        imgFCAC = cv.adaptiveThreshold(imgFCAC, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 55, 12)

        cv.imwrite('_amount\\A%s.jpg' % stri, imgAmount)
        cv.imwrite('_FCAC\\F%s.jpg' % stri, imgFCAC)
