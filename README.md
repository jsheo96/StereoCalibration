StereoDepthEstimation

This repository requires underwater_calibration directory which contains stereo images of checkerboard.

Users can sample the best checkerboard images by blurryness metric.
Stereo calibration and rectification are conducted from those samples.
Depths are estimated using rectified image and calibration matrix.

Main result of this repo is a calibration matrix(calibration.npz) and rectification matrix(params.xml)

Samples the best checkerboard images. This takes long time.
python find_focused_checkerboard.py

Creates calibration matrix from checkerboard images seen from stereo camera.
python low_cost_stereo_camera.py 

Manually tests the accuracy of depth estimation!
python manual_length_test.py
