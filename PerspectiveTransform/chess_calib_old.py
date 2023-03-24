# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import glob

# # prepare object points
# #Enter the number of inside corners in x
# nx = 9
# #Enter the number of inside corners in y
# ny = 6
# # Make a list of calibration images
# # chess_images = glob.glob('./camera_cal/cal*.jpg')
# # Select any index to grab an image from the list

# # Read in the image
# chess_board_image = mpimg.imread('./chess/kFM1C.jpg')
# # Convert to grayscale
# gray = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
# # Find the chessboard corners
# ret, corners = cv2.findChessboardCorners(chess_board_image, (nx, ny), None)
# # If found, draw corners
# if ret == True:
#     # Draw and display the corners
#     cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
#     result_name = 'board'+str(i)+'.jpg'
#     cv2.imshow(result_name, chess_board_image)
# else: 
#     print("Chessboard Not Found")

import cv2
import numpy as np
from argparse import ArgumentParser

# Load the image
img = cv2.imread("chess/tort_chess.jpg")

# img = cv2.imread("chess/test_image2.png")
# Color-segmentation to get binary mask
lwr = np.array([0, 0, 143])
upr = np.array([179, 61, 252])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
msk = cv2.inRange(hsv, lwr, upr)

# Extract chess-board
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
dlt = cv2.dilate(msk, krn, iterations=3)
res = 255 - cv2.bitwise_and(dlt, msk)
cv2.namedWindow('msk', cv2.WINDOW_NORMAL)
cv2.imshow('msk',msk)
cv2.namedWindow('krn', cv2.WINDOW_NORMAL)
cv2.imshow('krn',krn)
cv2.namedWindow('dlt', cv2.WINDOW_NORMAL)
cv2.imshow('dlt',dlt)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying chess-board features
res = np.uint8(res)
ret, corners = cv2.findChessboardCorners(res, (7, 7), None)
#  flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
#        cv2.CALIB_CB_FAST_CHECK +
#        cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret:
    print(corners.shape)
    fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
    cv2.imshow("fnl", fnl)
    cv2.waitKey(0)
else:
    print("No Checkerboard Found")