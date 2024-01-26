# Reference: https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
import cv2
import numpy as np
import tqdm
import pickle
import os
from scipy import stats


pathL = "underwater_calibration_focused/left/"
pathR = "underwater_calibration_focused/right/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((11 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 2 # cause the length of each box is 2 centemeters

img_ptsL = []
img_ptsR = []
obj_pts = []


def filter_outliers(points):
    # Calculate z-scores of each point
    z_scores = stats.zscore(points)

    # Set threshold for filtering outliers
    threshold = 2
    # Filter out points with a z-score outside of the threshold
    filtered_points = [point for i, point in enumerate(points) if abs(z_scores[i][0]) < threshold]
    filtered_points = np.array(filtered_points)
    return filtered_points

def detect_chessboard(path):
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=0.25,fy=0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 31, 31, 0.04)
    corners = cv2.dilate(corners, None)
    threshold = np.max(corners) * 0.2
    corner_points = np.argwhere(corners > threshold)
    corner_points = filter_outliers(corner_points)
    result = gray
    for corner_point in corner_points:
        i,j = corner_point
        result = cv2.circle(result, (j,i), 5, (0,255,0), -1)
    margin = 20
    minx = np.min(corner_points[:,1]) - margin
    maxx = np.max(corner_points[:,1]) + margin
    miny = np.min(corner_points[:,0]) - margin
    maxy = np.max(corner_points[:,0]) + margin
    result = cv2.rectangle(result, (minx,miny), (maxx,maxy), (255,255,255), 5)
    cv2.imshow('', result)
    cv2.waitKey(1)
    return minx * 4, miny * 4, maxx * 4, maxy * 4

for fn in tqdm.tqdm(os.listdir(pathL)):
    imgL = cv2.imread(pathL + fn)
    imgR = cv2.imread(pathR + fn)
    imgL_gray = cv2.imread(pathL + fn, 0)
    imgR_gray = cv2.imread(pathR + fn, 0)
    outputL = imgL.copy()
    outputR = imgR.copy()

    # masking checkerboard image to detect corners only inside checkerboard
    minxL, minyL, maxxL, maxyL = detect_chessboard(pathL + fn)
    minxR, minyR, maxxR, maxyR = detect_chessboard(pathR + fn)

    maskL = np.zeros_like(outputL)
    maskL[minyL:maxyL,minxL:maxxL, :] = 1
    outputL = outputL * maskL

    maskR = np.zeros_like(outputR)
    maskR[minyR:maxyR, minxR:maxxR, :] = 1

    retR, cornersR = cv2.findChessboardCorners(outputR, (11, 8), None)
    retL, cornersL = cv2.findChessboardCorners(outputL, (11, 8), None)

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria) # 88 1 2
        cv2.drawChessboardCorners(outputR, (11, 8), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (11, 8), cornersL, retL)
        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)
        print(fn)

# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

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
rectify_scale = 1
# pickle.dump([new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale,(0,0)], open('rectify_parameters_230324.pkl','wb'))

rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale,(0,0))
# pickle.dump(Q, open('Q_230324.pkl','wb'))
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_32FC1)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_32FC1)
# with open('left_inverse_rectification_parameters_230324.pkl', 'wb') as f:
#     pickle.dump([new_mtxL, distL, rect_l, proj_mat_l], f)
# with open('right_inverse_rectification_parameters_230324.pkl', 'wb') as f:
#     pickle.dump([new_mtxR, distR, rect_r, proj_mat_r], f)
print("Saving parameters ......")
cv_file = cv2.FileStorage("params_231206.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
print("Loading parameters ......")
cv_file = cv2.FileStorage("params_231206.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map = cv_file.getNode("Left_Stereo_Map_x").mat(),cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map = cv_file.getNode("Right_Stereo_Map_x").mat(),cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

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