import cv2
import numpy as np
import time
import os
import pickle

def get_remapped_coord(point, inverse_rectification_parameters):
    mtx, dist, R, new_mtx = inverse_rectification_parameters
    x,y = point
    return cv2.undistortPoints(np.array([[[x, y]]]).astype(np.float32), mtx, dist, R=R, P=new_mtx)[0][0]


if __name__ == '__main__':
    # left_folder = 'C:/Records/Local Records/underwater_calibration/left/'
    # right_folder = 'C:/Records/Local Records/underwater_calibration/right/'
    left_folder = r'\\wsl.localhost\Ubuntu\home\jsheo\FishSenseCore\images\left\\'
    right_folder = r'\\wsl.localhost\Ubuntu\home\jsheo\FishSenseCore\images\right\\'


    file_number = 158
    max_number = len(os.listdir(left_folder))
    fn = os.listdir(left_folder)[file_number]
    fnR = os.listdir(right_folder)[file_number]
    print(fn, fnR)
    imgL = cv2.imread(left_folder + fn)
    imgR = cv2.imread(right_folder + fnR)
    h,w,_ = imgL.shape

    # rectify image
    points = []
    points_label = ['headL', 'headR', 'tailL', 'tailR']
    scale = 3
    print('Click the head of fish on left image.')

    def get_length(points):
        start = time.time()
        head_left = points[0]
        head_right = points[1]
        tail_left = points[2]
        tail_right = points[3]

        # Rectify
        with open('left_inverse_rectification_parameters_230324.pkl','rb') as f:
            left_inverse_rectification_parameters = pickle.load(f)
        with open('right_inverse_rectification_parameters_230324.pkl','rb') as f:
            right_inverse_rectification_parameters = pickle.load(f)
        head_left_rectified = get_remapped_coord(head_left, left_inverse_rectification_parameters)
        head_right_rectified = get_remapped_coord(head_right, right_inverse_rectification_parameters)
        tail_left_rectified = get_remapped_coord(tail_left, left_inverse_rectification_parameters)
        tail_right_rectified = get_remapped_coord(tail_right, right_inverse_rectification_parameters)
        print('before rectification: {}, after rectification: {}'.format(head_left, head_left_rectified))

        head_disparity = head_left_rectified[0] - head_right_rectified[0]
        tail_disparity = tail_left_rectified[0] - tail_right_rectified[0]
        head_point = [head_left_rectified[0], head_left_rectified[1], head_disparity, 1]
        tail_point = [tail_left_rectified[0], tail_left_rectified[1], tail_disparity, 1]

        # 2D to 3D
        with open('Q_230324.pkl', 'rb') as f:
            Q = pickle.load(f)
        head_point_3d = np.matmul(Q, head_point)
        tail_point_3d = np.matmul(Q, tail_point)
        head_point_3d = (head_point_3d / head_point_3d[3])[:3]
        tail_point_3d = (tail_point_3d / tail_point_3d[3])[:3]
        print('head_point_3d: {}, tail_point_3d: {}'.format(head_point_3d, tail_point_3d))
        length = np.linalg.norm(head_point_3d - tail_point_3d)
        print(length)
        print('time for getting length', time.time() - start)
        return length

    def draw_circle(event, x, y, flags, param):
        global mouseX, mouseY, points, points_label
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y
            mouseX *= scale
            mouseY *= scale

            if len(points) % 2 == 1 and mouseX >= w: # When right image is clicked
                mouseX -= w
                cv2.circle(imgR, (mouseX, mouseY), 10, (255, 0, 0), -1)
                points.append((mouseX, mouseY))
                print('{}: {}'.format(points_label[len(points) - 1], points[-1]))
            elif len(points) % 2 == 0 and mouseX < w:
                cv2.circle(imgL, (mouseX, mouseY), 10, (255, 0, 0), -1)
                points.append((mouseX, mouseY))
                print('{}: {}'.format(points_label[len(points) - 1], points[-1]))

            if len(points) == 4:
                length = get_length(points)
                orgL = ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2)
                orgR = ((points[1][0]+points[3][0])//2, (points[1][1]+points[3][1])//2)

                cv2.putText(imgL, '{:.2f}'.format(length), orgL, 1, 8, (255,0,0), cv2.LINE_AA)
                cv2.putText(imgR, '{:.2f}'.format(length), orgR, 1, 8, (255,0,0), cv2.LINE_AA)
                points = [] # initialize


    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('test', draw_circle)
    while True:
        testImg = cv2.resize(np.hstack((imgL, imgR)), None, fx=1/scale, fy=1/scale)
        cv2.imshow('test', testImg)
        c = cv2.waitKey(10)
        if c == ord('e'):
            file_number += 1
            file_number %= max_number
            fn = os.listdir(left_folder)[file_number]
            fnR = os.listdir(right_folder)[file_number]
            imgL = cv2.imread(left_folder + fn)
            imgR = cv2.imread(right_folder + fnR)
        elif c == ord('q'):
            file_number -= 1
            fn = os.listdir(left_folder)[file_number]
            fnR = os.listdir(right_folder)[file_number]
            imgL = cv2.imread(left_folder + fn)
            imgR = cv2.imread(right_folder + fnR)
