import cv2
import numpy as np
from tqdm import tqdm
import pickle
import os
# Set the path to the images captured by the left and right cameras
pathL = "./underwater_calibration/left/"
pathR = "./underwater_calibration/right/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((11 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
# objectPoint 를 이렇게 meshgrid로 잡는 게 일반적인가?

img_ptsL = []
img_ptsR = []
obj_pts = []

calibration_focused_folder = 'underwater_calibration_focused/left'
for fn in os.listdir(calibration_focused_folder):
    imgL = cv2.imread(pathL + fn)
    imgR = cv2.imread(pathR + fn)
    imgL_gray = cv2.imread(pathL + fn, 0)
    imgR_gray = cv2.imread(pathR + fn, 0)
    # if os.path.exists('img_ptsL.pkl') and os.path.exists('img_ptsR.pkl') and os.path.exists('obj_pts.pkl'):
    #     break
    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, (11, 8), None)
    retL, cornersL = cv2.findChessboardCorners(outputL, (11, 8), None)
    if retR and retL:
        # print(objp)
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria) # 88 1 2

        cv2.drawChessboardCorners(outputR, (11, 8), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (11, 8), cornersL, retL)
        # cv2.imshow('cornersR', cv2.resize(outputR, None, fx=0.25, fy=0.25))
        # cv2.imshow('cornersL', cv2.resize(outputL, None, fx=0.25, fy=0.25))
        # cv2.waitKey()
        cv2.imwrite(os.path.join('underwater_calibration_chessboard/left', fn), outputR)
        cv2.imwrite(os.path.join('underwater_calibration_chessboard/right', fn), outputL)
        # exit()
        # print(i)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
        break
# pickle.dump(img_ptsL, open('img_ptsL_20.pkl','wb'))
# pickle.dump(img_ptsR, open('img_ptsR_20.pkl','wb'))
# pickle.dump(obj_pts, open('obj_pts_20.pkl','wb'))
img_ptsL = pickle.load(open('img_ptsL_20.pkl','rb'))
img_ptsR = pickle.load(open('img_ptsR_20.pkl','rb'))
obj_pts = pickle.load(open('obj_pts_20.pkl','rb'))
# print(len(obj_pts))
# exit()
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))


dst = cv2.undistort(imgL_gray, mtxL, distL, None, new_mtxL)
x,y,w,h = roiL
dst = dst[y:y + h, x:x + w]
cv2.imshow('dst', cv2.resize(dst, None, fx=0.25, fy=0.25))
cv2.waitKey()
# 각 카메라의 undistort 는 잘 되는 것으로 보아 그 아래 부분이 문제다.



flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR,
                                                                                    mtxL, distL, mtxR, distR,
                                                                                    imgL_gray.shape[::-1],
                                                                                    criteria_stereo, flags)
# flag에 관계 없이 똑같은데?
# flags, criteria 순서가 바뀐게 문제일 수도?
# before calibration 까지는 괜찮x
print(new_mtxL)
rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale,(0,0))

Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_16SC2)

# print("Saving parameters ......")
# cv_file = cv2.FileStorage("params_20.xml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
# cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
# cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
# cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
# cv_file.release()
# print("Loading parameters ......")
# cv_file = cv2.FileStorage("params_20.xml", cv2.FILE_STORAGE_READ)
# Left_Stereo_Map = cv_file.getNode("Left_Stereo_Map_x").mat(),cv_file.getNode("Left_Stereo_Map_y").mat()
# Right_Stereo_Map = cv_file.getNode("Right_Stereo_Map_x").mat(),cv_file.getNode("Right_Stereo_Map_y").mat()
# cv_file.release()

cv2.imshow("Left image before rectification", cv2.resize(imgL, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR))
cv2.imshow("Right image before rectification", cv2.resize(imgR, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR))

Left_nice = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_nice = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
print(Left_nice.shape, Right_nice.shape)
cv2.imshow("Left image after rectification", cv2.resize(Left_nice, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR))
cv2.imshow("Right image after rectification", cv2.resize(Right_nice, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR))
cv2.waitKey(0)

out = Right_nice.copy()
out[:, :, 0] = Right_nice[:, :, 0]
out[:, :, 1] = Right_nice[:, :, 1]
out[:, :, 2] = Left_nice[:, :, 2]

cv2.imshow("Output image", cv2.resize(out, None, fx=0.25, fy=0.25))
cv2.waitKey(0)