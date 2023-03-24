import cv2
import numpy as np
from argparse import ArgumentParser
from datetime import datetime  
import sys



def get_parser():
    '''
    --img1 : thermal image (keyhole) 
    --img2 : visible image (key)
    --output : path of output image(blend of distort img2 and img1)
    --width  : width of output
    --height : height of output
    --byhand : wether to choose points by hand(mouse)
    '''
    dt = datetime.now()
    ts = datetime.timestamp(dt)

    parser = ArgumentParser(description='my description')
    parser.add_argument('--img1', type=str, default="test_0111/FLIR0100.jpg", help="img1 for thermal image")
    parser.add_argument('--img2', type=str, default="test_0111/FLIR0101.jpg", help="img2 for visible image")
    parser.add_argument('--output', type=str, default="test_result/result.jpg")
    parser.add_argument('--width', type=int, default=640, help="crop width, crop center")
    parser.add_argument('--height', type=int, default=480, help="crop height, crop center")
    parser.add_argument('--byhand', action='store_true')
    parser.add_argument('--autowh', action='store_true')

    return parser

class DrawPointWidget(object):

    def __init__(self, img, is_vis):
        self.image = img
        self.clone = self.image.copy()
        cv2.namedWindow('image')
        # check if image is visible image
        if is_vis:
            cv2.setMouseCallback('image', self.draw_visible)
        else:
            cv2.setMouseCallback('image', self.draw_thermal)

    def draw_visible(self, event, x, y, flags, parameters):
        global coordinates, coordinates_start # coordinates and coordinates_start = (-1,-1)
        global points

        # Click Mbutton to choose points and line
        if event == cv2.EVENT_MBUTTONDOWN:
            # draw circle of 2px
            print("Mbutton down")
            cv2.circle(self.clone, (x,y), 3, (0, 255, 0), -1)
            cv2.imshow("image", self.clone) 

            if coordinates[0] != -1: # if coordinates are not first points, then draw a line
                cv2.line(self.clone, coordinates, (x, y), (36, 255,12), 2)
                cv2.imshow("image", self.clone) 
            else: # if coordinates are first points, store as starting points
                coordinates_start = (x, y)
                cv2.imshow("image", self.clone) 
            coordinates = (x, y)
            points.append(coordinates)
            
        # Double click Lbutton to connect start and end points
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print("Lbutton dbclick")
            cv2.line(self.clone, coordinates, coordinates_start, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("image", self.clone) 

            coordinates = (-1, -1) # reset ix and iy
        
        # Click Rbutton to reset points
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Rbutton down")
            self.clone = self.image.copy()
            cv2.imshow("image", self.clone) 
            points = []
            coordinates = (-1, -1)

    # same as draw 
    def draw_thermal(self, event, x, y, flags, parameters):
        global coordinates, coordinates_start, points
        
        if event == cv2.EVENT_MBUTTONDOWN:
            # draw circle of 2px
            print("Mbutton down")
            cv2.circle(self.clone, (x,y), 3, (100, 100, 100), -1)
            cv2.imshow("image", self.clone) 

            if coordinates[0] != -1: # if ix and iy are not first points, then draw a line
                cv2.line(self.clone, coordinates, (x, y), (100, 100, 100), 2)
                cv2.imshow("image", self.clone) 
            else: # if ix and iy are first points, store as starting points
                coordinates_start = (x, y)
                cv2.imshow("image", self.clone) 
            coordinates = (x, y)
            points.append(coordinates)
            
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print("Lbutton dbclick")
            # if flags == 33: # if alt key is pressed, create line between start and end points to create polygon
            cv2.line(self.clone, coordinates, coordinates_start, (100, 100, 100), 2, cv2.LINE_AA)
            cv2.imshow("image", self.clone) 

            coordinates = (-1, -1) # reset ix and iy
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Rbutton down")
            self.clone = self.image.copy()
            cv2.imshow("image", self.clone) 
            coordinates = (-1, -1)
            points = []
            
    def show_image(self):
        return self.clone

if __name__=="__main__":
    ### Get parser
    parser = get_parser()
    args = parser.parse_args()
    img_ther = cv2.imread(args.img1) # size=(320,240)
    img_vis = cv2.imread(args.img2) # size=(640,480)

    if args.autowh:
        print("autowh")
        width, height = img_vis.shape[1], img_vis.shape[0]
    else:
        width, height = args.width, args.height
    img_ther = cv2.resize(img_ther, (width,height), interpolation=cv2.INTER_AREA)

    # img_vis = img_vis[88:88+384, 52:52+512]
    # img_vis_sm = cv2.resize(img_vis, (width,height), None)
    img_blend = cv2.addWeighted(img_vis, 0.4, img_ther, 0.8, 0.0)
    cv2.imwrite("test_result/normal_blend.jpg", img_blend)
    
    ### Get points
    points = []
    draw_vis_widget = DrawPointWidget(img_vis, is_vis=True)
    coordinates = (-1,-1)
    coordinates_start = (-1,-1)

    while True:
        cv2.imshow('image', draw_vis_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if (key == ord('q') or key == ord(' ')):
            cv2.destroyAllWindows()
            # cv2.imwrite("drawline_ch05.jpg", draw_point_widget.show_image())
            break

    img_ther = cv2.resize(img_ther.copy(), (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_AREA)
    ther_clone = img_ther.copy()
    vis_points = points.copy()
    coordinates = (-1,-1)
    coordinates_start = (-1,-1)
    # draw rect of the visible image
    start_p = points[0]
    for i in range(len(points)-1):
        cv2.line(ther_clone, points[i],points[i+1],(36, 255,12), 2)
    cv2.line(ther_clone, points[-1], start_p,(36, 255,12), 2)
    points = []
    draw_ther_widget = DrawPointWidget(ther_clone, is_vis=False)
    

    while True:
        cv2.imshow('image', draw_ther_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if (key == ord('q') or key == ord(' ')):
            cv2.destroyAllWindows()
            break
    ther_points = points.copy()
    print(vis_points)
    print(ther_points)

    # print('exit')
    # sys.exit()
    
    # img_vis = cv2.resize(img_vis, (320,240), interpolation = cv2.INTER_CUBIC)
    
    if args.byhand:
        # original pts
        pts_o = np.float32(vis_points) # 这四个点为原始图片上数独的位置
        pts_d = np.float32(ther_points) # 这是变换之后的图上四个点的位置
    else:
        pts_o = np.float32([[48, 164], [60, 238], [184, 194], [146, 131]]) # 这四个点为原始图片上数独的位置
        pts_d = np.float32([[36, 96], [43, 146], [123, 117], [98, 78]]) # 这是变换之后的图上四个点的位置
        # pts_o = np.float32([[36, 1043], [905, 1041], [903, 166], [34, 181]]) # 这四个点为原始图片上数独的位置
        # pts_d = np.float32([[29, 1049], [792, 1047], [924, 158], [122, 176]]) # 这是变换之后的图上四个点的位置

    # get transform matrix
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # apply transformation
    dst = cv2.warpPerspective(img_vis, M, (width, height)) # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定
    dst2 = cv2.addWeighted(dst, 0.8, img_ther, 0.8, 0.0)

    # cv2.imshow('img_ther', img_ther)
    # cv2.imshow('im2', img_vis)
    cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    cv2.imshow('dst', dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(args.output, dst)
    cv2.imwrite("test_result/blend_0209.jpg", dst2)
