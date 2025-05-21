import cv2
import numpy as np
import time
import os

# Parameters (should match the 1 generation script)

# for charuco_watch.mov use e.g. DICT_5X5_100, DICT_5X5_250 or DICT_5X5_1000, 
# but correct square_length and marker_length unknown (thus no board pose estimation possible -> interpolateCornersCharuco fails)
# ARUCO_DICT = cv2.aruco.DICT_5X5_250

# for data captured by William
ARUCO_DICT = cv2.aruco.DICT_6X6_250 

SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015

# Input source path
INPUT_SOURCE = "data/clean/VID20250521123919.mp4" 

USE_CALIBRATION = True

# William's Oneplus 12 Smartphone (Photos)
# DIST_COEFFS = np.array([0.360103, -3.711869, 0.000724, -0.002625, 11.356215])
# CAMERA_MATRIX = np.array([
#     [2831.81, 0.00, 2049.37],
#     [0.00, 2776.26, 1468.23],
#     [0.00, 0.00, 1.00]
# ])

# William's Oneplus 12 Smartphone (Video)
DIST_COEFFS = np.array([0.067710, -0.643980, 0.000172, 0.000833, -0.935000])
CAMERA_MATRIX = np.array([
    [3360.11, 0.00, 1934.39],
    [0.00, 3297.00, 893.22],
    [0.00, 0.00, 1.00]
])

def setup_charuco_board():
    """Initialize the ChArUco board with the same parameters used for generation."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), 
                                 SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    return board, dictionary



def detect_charuco(frame, board, dictionary, camera_matrix, dist_coeffs, use_calibration):
    """Detect ChArUco markers with camera calibration and pose estimation."""
    # Scale down the frame to 50%
    scale_percent = 100
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    image_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    aruco_params = cv2.aruco.DetectorParameters()
    # Use subpixel refinement for more accurate corner detection
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Parameters for adaptive thresholding (used to binarize the image)
    # Increased window sizes and constant for better handling of motion blur
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 70  # Increased from 45 for better blur handling
    aruco_params.adaptiveThreshWinSizeStep = 5
    aruco_params.adaptiveThreshConstant = 9  # Increased from 7 for better contrast handling
    
    # Parameters for marker detection
    # Slightly relaxed parameters for motion blur
    aruco_params.minMarkerPerimeterRate = 0.08  # Slightly decreased from 0.1
    aruco_params.maxMarkerPerimeterRate = 2.0
    aruco_params.polygonalApproxAccuracyRate = 0.03  # Slightly increased from 0.02
    aruco_params.minCornerDistanceRate = 0.1
    aruco_params.minMarkerDistanceRate = 0.1
    aruco_params.minDistanceToBorder = 3
    
    # Parameters for marker quality
    # Reduced contrast requirements for motion blur
    aruco_params.minOtsuStdDev = 3.5  # Decreased from 5.0
    # Increased cell size for better blur handling
    aruco_params.perspectiveRemovePixelPerCell = 6  # Increased from 4
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    
    # Parameters for error correction
    # Increased error tolerance for motion blur
    aruco_params.maxErroneousBitsInBorderRate = 0.4  # Increased from 0.35
    aruco_params.errorCorrectionRate = 0.75  # Increased from 0.6

    detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Debug information
    print(f"Number of markers detected: {len(corners) if corners is not None else 0}")        
    print(f"Marker IDs detected: {ids.flatten() if ids is not None else 0}")
    
    if ids is not None and len(ids) > 0:
        # Draw detected/rejected markers
        cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
        # if rejected is not None and len(rejected) > 0:
        #     cv2.aruco.drawDetectedMarkers(image_copy, rejected, None, (0, 0, 255))
        
        # Detect ChArUco corners
        if use_calibration:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, frame, board,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs)
        else:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, frame, board)
        
        # Debug information
        print(f"Number of ChArUco corners detected: {retval}")
        print(f"ChArUco corner IDs: {charuco_ids.flatten() if charuco_ids is not None else 0}")
        
        if use_calibration:
            if retval > 0 and charuco_ids is not None and len(charuco_ids) > 0:
                # Draw detected ChArUco corners
                cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids, (255, 0, 0))
                
                # Initialize rotation and translation vectors
                rvec = np.zeros(3, dtype=np.float32)
                tvec = np.zeros(3, dtype=np.float32)
                
                # Estimate pose
                valid = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, rvec, tvec)
                
                if valid:
                    # Draw coordinate axes
                    cv2.drawFrameAxes(image_copy, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    print(f"Translation: {tvec.flatten()}")
                    print(f"Rotation (degrees): {np.degrees(rvec.flatten())}")
    
    return image_copy



def process_image(image_path, board, dictionary, use_calibration=False):
    """Process a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    frame = detect_charuco(frame, board, dictionary, CAMERA_MATRIX, DIST_COEFFS, use_calibration)
    
    
    # Display the frame
    cv2.imshow("ChArUco Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, board, dictionary, use_calibration=False):
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_count += 1
        print(f"\nProcessing frame {frame_count}")
        
        # Detect ChArUco markers
        frame = detect_charuco(frame, board, dictionary, CAMERA_MATRIX, DIST_COEFFS, use_calibration)
        
        # Display the frame
        cv2.imshow("ChArUco Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Setup ChArUco board
    board, dictionary = setup_charuco_board()
    
    # Get absolute path of the input source
    input_path = os.path.abspath(INPUT_SOURCE)
    
    # Check if file exists
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    # Check file extension
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"Processing image: {input_path}")
        process_image(input_path, board, dictionary, USE_CALIBRATION)
    elif ext in ['.mp4', '.avi', '.mov']:
        print(f"Processing video: {input_path}")
        process_video(input_path, board, dictionary, USE_CALIBRATION)
    else:
        print(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    main()
