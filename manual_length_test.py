import cv2
import numpy as np
import os

if __name__ == '__main__':
    file_number = 1000
    max_number = len(os.listdir('underwater_calibration/left'))
    imgL = cv2.imread('underwater_calibration/left/{:05d}.png'.format(file_number))
    imgR = cv2.imread('underwater_calibration/right/{:05d}.png'.format(file_number))
    h,w,_ = imgL.shape

    # rectify image
    print("Loading parameters ......")
    cv_file = cv2.FileStorage("params_20.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map = cv_file.getNode("Left_Stereo_Map_x").mat(),cv_file.getNode("Left_Stereo_Map_y").mat()
    Right_Stereo_Map = cv_file.getNode("Right_Stereo_Map_x").mat(),cv_file.getNode("Right_Stereo_Map_y").mat()
    cv_file.release()
    imgL = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    points = []
    points_label = ['headL', 'headR', 'tailL', 'tailR']
    scale = 4
    print('Click the head of fish on left image.')
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
                Bf = 428.742112
                head_disparity = abs(points[0][0] - points[1][0])
                tail_disparity = abs(points[2][0] - points[3][0])
                head_depth = Bf / head_disparity
                tail_depth = Bf / tail_disparity
                diff_depth = abs(tail_depth - head_depth)
                cx = 1.27553668e+03
                cy = 9.83886486e+02
                fx = 2.14371056e+03
                fy = 2.15009239e+03
                head_x = (points[0][0] - cx) / fx * head_depth
                tail_x = (points[2][0] - cx) / fx * tail_depth
                diff_x = abs(head_x - tail_x)

                head_y = (points[0][1] - cy) / fy * head_depth
                tail_y = (points[2][1] - cy) / fy * tail_depth
                diff_y = abs(head_y - tail_y)
                length = np.sqrt((diff_x ** 2 + diff_y ** 2 + diff_depth ** 2))
                print('distance to head: {} cm'.format(head_depth*100))
                print('distance to tail: {} cm'.format(tail_depth*100))
                print('fork length: {} cm'.format(length * 100))

                orgL = ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2)
                orgR = ((points[1][0]+points[3][0])//2, (points[1][1]+points[3][1])//2)

                cv2.putText(imgL, '{:.2f}'.format(length*100), orgL, 1, 8, (255,0,0), cv2.LINE_AA)
                cv2.putText(imgR, '{:.2f}'.format(length*100), orgR, 1, 8, (255,0,0), cv2.LINE_AA)
                points = [] # initialize



    cv2.namedWindow('test')
    cv2.setMouseCallback('test', draw_circle)
    while True:
        testImg = cv2.resize(np.hstack((imgL, imgR)), None, fx=1/scale, fy=1/scale)
        cv2.imshow('test', testImg)
        c = cv2.waitKey(10)
        if c == ord('e'):
            file_number += 1
            file_number %= max_number
            imgL = cv2.imread('underwater_calibration/left/{:05d}.png'.format(file_number))
            imgR = cv2.imread('underwater_calibration/right/{:05d}.png'.format(file_number))
            imgL = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            imgR = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        elif c == ord('q'):
            file_number -= 1
            file_number %= max_number
            imgL = cv2.imread('underwater_calibration/left/{:05d}.png'.format(file_number))
            imgR = cv2.imread('underwater_calibration/right/{:05d}.png'.format(file_number))
            imgL = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            imgR = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
