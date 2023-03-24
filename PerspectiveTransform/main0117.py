## find four corner point of a card to do transformation
import argparse
import cv2
import numpy as np
from functions0117 import four_point_transform, crop

binary_threshold = 128
min_area = 100

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--img1", type=str, default='img1.jpg', help="path for the object image")
    # parser.add_argument("--img2", type=str, default='img2.jpg', help="path for image containing the object")
    # parser.add_argument("--h", type=int, default=0, help="how you want img1 to crop")
    
    # args = parser.parse_args()

    # img1 = cv2.imread(args.img1)
    # img2 = cv2.imread(args.img2)
    # img1 = crop(img1, args.h)
    img = cv2.imread('test_0111/joker.jpg')
    # blur
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    ret, thresh = cv2.threshold(gray_img, binary_threshold, 255, cv2.THRESH_BINARY)
    print(thresh.shape)
    kernel = np.array((7,7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("a", thresh)
    cv2.waitKey()

    # Find contours in binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours))
    contours_list = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        # print(area)
        if area > min_area:
            contours_list.append(contours[i])
    contours = tuple(contours_list)
    print( "there are " + str(len(contours)) + " contours")
    cnt = contours[0]
    print ("there are " + str(len(cnt)) + " points in contours[0]")
    img_cnt = img.copy()
    cv2.drawContours(img_cnt,contours,-1,(0,0,255),thickness=1,lineType=cv2.LINE_4)  # AA is better but more
    cv2.imshow('img', img_cnt)
    cv2.waitKey(0)

    peri = cv2.arcLength(cnt, True)
    corners = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    print(corners)
    result = img.copy()
    cv2.polylines(result, [corners], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("QUAD", result)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    pts = np.array(eval(args["coords"]), dtype = "float32")
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)
    # show the original and warped images
    cv2.imshow("Original", image)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
