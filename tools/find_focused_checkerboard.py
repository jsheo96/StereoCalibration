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

for i in tqdm(range(0, 2337)):
    imgL = cv2.imread(pathL + "%05d.png" % i)
    imgR = cv2.imread(pathR + "%05d.png" % i)
    imgL_gray = cv2.imread(pathL + "%05d.png" % i, 0)
    imgR_gray = cv2.imread(pathR + "%05d.png" % i, 0)
    # if os.path.exists('img_ptsL.pkl') and os.path.exists('img_ptsR.pkl') and os.path.exists('obj_pts.pkl'):
    #     break
    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, (11, 8), None)
    retL, cornersL = cv2.findChessboardCorners(outputL, (11, 8), None)
    if retL and retR:
        minxL = np.min(cornersL[:,:,0])
        maxxL = np.max(cornersL[:,:,0])
        minyL = np.min(cornersL[:, :, 1])
        maxyL = np.max(cornersL[:, :, 1])

        minxR = np.min(cornersR[:,:,0])
        maxxR = np.max(cornersR[:,:,0])
        minyR = np.min(cornersR[:, :, 1])
        maxyR = np.max(cornersR[:, :, 1])

        roiL = outputL[int(minyL):int(maxyL), int(minxL):int(maxxL), :]
        roiR = outputR[int(minyR):int(maxyR), int(minxR):int(maxxR), :]
        # cv2.imshow('1',roiL)
        # cv2.imshow('2',roiR)
        # cv2.waitKey()


        fmL = cv2.Laplacian(roiL, cv2.CV_64F).var()
        fmR = cv2.Laplacian(roiR, cv2.CV_64F).var()
        string ='{} {} {}\n'.format(i, fmL,fmR)
        with open('blur_metric.txt','a') as f:
            f.write(string)
