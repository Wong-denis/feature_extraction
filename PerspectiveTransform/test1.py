import argparse
import os
import cv2
import numpy as np

def get_corrected_img(img1, img2):
    MIN_MATCHES = 10

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(len(good_matches))
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img
    return img2

def crop(img, crop_h):
    crop_img = img
    
    if len(img.shape) > 2:
        h, w, chn = img.shape
    else:
        h, w = img.shape
    
    if crop_h:
        crop_w = w
        # start_x = (h - crop_h)//2 # align center
        # start_y = (w - crop_w)//2 # align center
        start_x = 0 # align top
        start_y = 0 # align left
        crop_img = img[start_x:start_x+crop_h, start_y:start_y+crop_w]
    
    return crop_img
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--img1", type=str, default='img1.jpg', help="path for the object image")
    parser.add_argument("--img2", type=str, default='img2.jpg', help="path for image containing the object")
    parser.add_argument("--h", type=int, default=0, help="how you want img1 to crop")
    
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    img1 = crop(img1, args.h)

    img = get_corrected_img(img2, img1)
    cv2.imshow('Original image', img1)
    cv2.imshow('Corrected image', img)
    cv2.waitKey(0)
    cv2.imwrite('test_result/orig01.jpg', img1)
    cv2.imwrite('test_result/result01.jpg',img)