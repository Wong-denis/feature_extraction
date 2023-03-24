import cv2
import numpy as np
from argparse import ArgumentParser

def get_parser():
    '''
    --img1 : dest image 
    --img2 : src image
    --output : path of output image(blend of distort img2 and img1)
    --width  : width of output
    --height : height of output
    --byhand : wether to choose points by hand(mouse)
    '''
    parser = ArgumentParser(description='my description')
    parser.add_argument('--img1', type=str, default="test_0111/FLIR0100.jpg")
    parser.add_argument('--img2', type=str, default="test_0111/FLIR0101.jpg")
    parser.add_argument('--output', type=str, default="test_result/result_0208.jpg")
    parser.add_argument('--width', type=int, default=320, help="crop width, crop center")
    parser.add_argument('--height', type=int, default=240, help="crop height, crop center")
    parser.add_argument('--byhand', action='store_true')
    
    return parser

### find coordinate of devices
def click_event(event,x,y,flags,params):
    
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x,'',y)
        points.append([x,y])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    img1 = cv2.imread(args.img1) # size=(320,240)
    img2 = cv2.imread(args.img2) # size=(640,480)
    # img2 = img2[88:88+384, 52:52+512]
    img3 = cv2.resize(img2, (320,240), None)
    img3 = cv2.addWeighted(img3, 0.4, img1, 0.8, 0.0)
    cv2.imwrite("test_result/normal_blend.jpg", img3)
    points = []

    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img1)
    image = img1
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    points_1 = points.copy()
    print(points_1)
    points = []

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img2)
    image = img2
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    points_2 = points.copy()
    print(points_2)
    print('=======================')

    
    cv2.destroyAllWindows()
    
    # img2 = cv2.resize(img2, (320,240), interpolation = cv2.INTER_CUBIC)
    
    if args.byhand:
        # original pts
        pts_o = np.float32(points_2) # 这四个点为原始图片上数独的位置
        pts_d = np.float32(points_1) # 这是变换之后的图上四个点的位置
    else:
        pts_o = np.float32([[48, 164], [60, 238], [184, 194], [146, 131]]) # 这四个点为原始图片上数独的位置
        pts_d = np.float32([[36, 96], [43, 146], [123, 117], [98, 78]]) # 这是变换之后的图上四个点的位置
        # pts_o = np.float32([[36, 1043], [905, 1041], [903, 166], [34, 181]]) # 这四个点为原始图片上数独的位置
        # pts_d = np.float32([[29, 1049], [792, 1047], [924, 158], [122, 176]]) # 这是变换之后的图上四个点的位置

    # get transform matrix
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # apply transformation
    dst = cv2.warpPerspective(img2, M, (args.width, args.height)) # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定
    dst2 = cv2.addWeighted(dst, 0.5, img1, 1, 0.0)

    # cv2.imshow('img1', img1)
    # cv2.imshow('im2', img2)
    cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    cv2.imshow('dst', dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(args.output, dst)
    cv2.imwrite("test_result/blend_0208.jpg", dst2)
