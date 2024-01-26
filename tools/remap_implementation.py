import cv2
import os
import numpy as np
import pickle

def get_remapped_coord(point, inverse_rectification_parameters):
    mtx, dist, R, new_mtx = inverse_rectification_parameters
    x,y = point
    return cv2.undistortPoints(np.array([[[x, y]]]).astype(np.float32), mtx, dist, R=R, P=new_mtx)[0][0]

if __name__ == '__main__':
    left_folder = 'C:/Records/Local Records/underwater_calibration/left/'
    right_folder = 'C:/Records/Local Records/underwater_calibration/right/'

    file_number = 0
    max_number = len(os.listdir(left_folder))
    fn = os.listdir(left_folder)[file_number]
    imgL = cv2.imread(left_folder + fn)

    print("Loading parameters ......")
    cv_file = cv2.FileStorage("params_230324.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map = cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat()
    cv_file.release()

    imgL_rectified1 = np.zeros_like(imgL)
    for i_rectified in range(imgL_rectified1.shape[0]):
        for j_rectified in range(imgL_rectified1.shape[1]):
            i = int(Left_Stereo_Map[1][i_rectified, j_rectified])
            j = int(Left_Stereo_Map[0][i_rectified, j_rectified])
            if i >= 0 and i < imgL.shape[0] and j >= 0 and j < imgL.shape[1]:
                imgL_rectified1[i_rectified, j_rectified, :] = imgL[i, j, :]
    imgL_rectified2 = np.zeros_like(imgL)
    with open('left_inverse_rectification_parameters_230324.pkl', 'rb') as f:
        inverse_rectification_parameters = pickle.load(f)
    for i in range(imgL.shape[0]):
        for j in range(imgL.shape[1]):
            point = (j, i)
            remapped_point = get_remapped_coord(point, inverse_rectification_parameters)
            j_rectified = int(remapped_point[0])
            i_rectified = int(remapped_point[1])
            if i_rectified >= 0 and i_rectified < imgL_rectified2.shape[0] and j_rectified >= 0 and j_rectified < imgL_rectified2.shape[1]:
                imgL_rectified2[i_rectified, j_rectified, :] = imgL[i, j, :]
    cv2.namedWindow('imgL_rectified', cv2.WINDOW_NORMAL)
    cv2.imshow('imgL_rectified', np.vstack((imgL_rectified1, imgL_rectified2)))
    cv2.waitKey(0)
