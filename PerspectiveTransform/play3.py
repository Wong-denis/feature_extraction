import numpy as np
import cv2 
from argparse import ArgumentParser
import os

# from matplotlib import pyplot as plt

# img1 = cv2.imread('./feat_match/upper.png',cv2.IMREAD_GRAYSCALE) # queryImage
# img2 = cv2.imread('./feat_match/lower.png',cv2.IMREAD_GRAYSCALE) # trainImage
# img1 = cv2.imread('./feat_match/far.png',cv2.IMREAD_GRAYSCALE) # queryImage
# img2 = cv2.imread('./feat_match/close.png',cv2.IMREAD_GRAYSCALE) # trainImage
def get_parser():
    parser = ArgumentParser(description='my description')
    parser.add_argument('--input1', type=str, default="", help='image or dir of images waiting for crop')
    parser.add_argument('--input2', type=str, default="", help='image or dir of images waiting for crop')
    parser.add_argument('--output', type=str, default="", help='where should the output image or images be store')
    # parser.add_argument('--width', type=int, default=0, help="crop width, crop center")
    # parser.add_argument('--height', type=int, default=0, help="crop height, crop center")
    return parser

parser = get_parser()
args = parser.parse_args()
# if len(args.input1.shape) > 2:
img1 = cv2.imread(args.input1,cv2.IMREAD_GRAYSCALE) # queryImage
img2 = cv2.imread(args.input2,cv2.IMREAD_GRAYSCALE) # trainImage
# cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

############# Enhance Contrast #############
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
# Top Hat Transform
topHat = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel)
# Black Hat Transform
blackHat = cv2.morphologyEx(img2, cv2.MORPH_BLACKHAT, kernel)
res = img2 + topHat - blackHat
# cv2.imshow("img2_contrast10", res)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
# Top Hat Transform
topHat = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kernel)
# Black Hat Transform
blackHat = cv2.morphologyEx(img2, cv2.MORPH_BLACKHAT, kernel)
res = img2.astype(int) + topHat.astype(int) - blackHat.astype(int)
new_res = np.square(res)*(-0.00136) + res*1.212
new_res[new_res > 255]  = 255
new_res[new_res < 0]    = 0
# over_res = res > 255
# part_res = res * over_res
# part_res -= 255
cv2.imshow("new_res", new_res.astype(np.uint8))
img2 = new_res.astype(np.uint8)
kernel2 = np.ones((7,7),np.uint8)
gradient = cv2.morphologyEx(img2, cv2.MORPH_GRADIENT, kernel2)
gradient[gradient > 90] = 90
gradient[gradient < 20] = 5
cv2.imshow("gradient", gradient)

img2 = (gradient.astype(int))*0.9 + (img2.astype(int))*0.9 
img2[img2 > 255]  = 255
img2 = img2.astype(np.uint8)
cv2.imshow("new_img2", img2)
cv2.waitKey(0)


# ##### check if there is pixel > 255 ; Some pixel will overflow and white turn black
# over_top = topHat > 35
# over_blk = blackHat > 30
# # print(np.unique(over)) 
# cv2.imshow("over_top", (over_top.astype(np.uint8))*255)
# cv2.imshow("over_blk", (over_blk.astype(np.uint8))*255)
# # cv2.imshow("top_hat", cv2.resize(topHat, (topHat.shape[1]*0.8, topHat.shape[0]*0.8)))
# # cv2.imshow("black_hat", cv2.resize(blackHat, (blackHat.shape[1]*0.8, blackHat.shape[0]*0.8)))
# cv2.imshow("img2_contrast35", res)
# cv2.waitKey(0)

cv2.destroyAllWindows()
############################################
# os._exit()

############# show feature points ###########
# Initiate SIFT detector
sift = cv2.ORB_create()
# find the keypoints with SIFT
kp = sift.detect(img1,None)
# compute the descriptors with SIFT
kp, des = sift.compute(img1, kp)
# draw only keypoints location,not size and orientation
imgResult = cv2.drawKeypoints(img1, kp, None, color=(0,255,0), flags=0)
cv2.imshow("img1", imgResult)
cv2.waitKey(0)
# plt.title('SIFT Algorithm for image 1')
# plt.imshow(imgResult)
# plt.show()

# Initiate SIFT detector
sift = cv2.ORB_create()
# find the keypoints with SIFT
kp = sift.detect(img2,None)
# compute the descriptors with SIFT
kp, des = sift.compute(img2, kp)
# draw only keypoints location,not size and orientation
imgResult = cv2.drawKeypoints(img2, kp, None, color=(0,255,0), flags=0)
cv2.imshow("img2", imgResult)
cv2.waitKey(0)
#############################################

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

########## Brute Force ##########
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("img3",img3)
cv2.waitKey(0)
cv2.imwrite("play_result/bf_result.png", img3)
###################################

########## FLANN ##########
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
print(len(matches))
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    # print(f"m,n: {m.distance},{n.distance}")
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow("img3",img3)
cv2.waitKey(0)
cv2.imwrite("play_result/flann_result.png", img3)
############################

MIN_MATCH_COUNT = 10
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print(img1.shape)

import timeit

start = timeit.default_timer()
 
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[h-1,w-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow("img4",img4)
cv2.waitKey(0)
cv2.imwrite(args.output, img4)

stop = timeit.default_timer()
waktu = stop - start
print('Time: ', stop - start) 

avg = sum(matchesMask)/len(matchesMask)
round(avg)
persentase = round(avg)*100

from tabulate import tabulate
table = [['Methods','Keypoint1-upper', 'Keypoint2-lower', 'Matches', "AVG Matches Rate","Time(sec)"], ["SIFT",len(kp1), len(kp2), len(matchesMask), persentase, waktu]]
print(tabulate(table, headers='firstrow'))