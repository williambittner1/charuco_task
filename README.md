# ChArUco Board Detection and Camera Calibration

This repository contains tools for working with ChArUco boards, including board generation, camera calibration and marker detection and board pose estimation using OpenCV.

## Features

- Generate custom ChArUco boards (1_generate_charuco.py)
- Camera calibration using ChArUco boards (2_calibrate_camera.py)
- Detect ChArUco markers in images and videos and pose estimation with calibrated cameras (3_detect_charuco.py)

## Requirements

```bash
conda create -n charuco python=3.11 opencv -c conda-forge -y
conda activate charuco
```

## Usage

### 1. Generate a ChArUco Board

```bash
python 1_generate_charuco.py
```

This script will: 
- create a ChArUco board image with the specified parameters.
- Save it to an image file

### 2. Camera Calibration

1. Set CALIBRATION_FOLDER (with multiple photos of the ChArUco board from different angles)
2. Run the calibration script:

```bash
python 2_calibrate_camera.py
```

This script will:
- Process each calibration image
- Detect ChArUco markers and corners
- Calculate camera matrix and distortion coefficients
- Save results to `camera_calibration.npz`
- Print out Camera Matrix and Distortion Coefficients

### 3. Detect ChArUco Markers

```bash
python 3_detect_charuco.py
```

This script will:
- Load an image or video
- Detect ChArUco markers
- Estimate pose (if calibration data is available)
- Display results in real-time

## Parameters

You can adjust the following parameters in the scripts:

- `ARUCO_DICT`: ArUco dictionary type
- `SQUARES_VERTICALLY`: Number of squares vertically
- `SQUARES_HORIZONTALLY`: Number of squares horizontally
- `SQUARE_LENGTH`: Physical length of squares (in meters)
- `MARKER_LENGTH`: Physical length of markers (in meters)


## Challenges:
- Choose correct Board ARUCO_DICT (e.g. 6X6_250). Otherwise, if the wrong board setup is being used during detection stage, the detection won't work
- Interpolate Charuco Corners needs correct/exact board parameters: e.g. SQUARE_LENGTH = 0.03, MARKER_LENGTH = 0.015 otherwise the corners won't get interpolated and no board pose gets predicted
- Problems with strong Motion Blur
- Pose Estimation is less accurate for videos than for images for some reason