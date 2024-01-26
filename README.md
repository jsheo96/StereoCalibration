# StereoDepthEstimation
This repository requires underwater_calibration directory which contains stereo images of checkerboard.

## Introduction
- Users can sample the best checkerboard images by blurryness metric.
- Stereo calibration and rectification are conducted from those samples.
- Depths are estimated using rectified image and calibration matrix.

## Folder structure
Image folder named "underwater_calibration" should have below structure.
```
underwater_calibration
├── left
└── right
```


## Result
Main result of this repo is a calibration matrix(calibration.npz) and rectification matrix(params.xml)

## Usage
Samples the best checkerboard images. This takes long time.
```
python find_focused_checkerboard.py 
```

Creates calibration matrix from checkerboard images seen from stereo camera.
```
python low_cost_stereo_camera.py 
```

Manually tests the accuracy of depth estimation!
```
python manual_length_test.py
```

## Reference
[https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)