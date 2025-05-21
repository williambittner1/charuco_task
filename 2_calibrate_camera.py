import cv2
import numpy as np
import glob
import os

# Parameters (should match the generation script)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.02

CALIBRATION_FOLDER = 'data/calibration'

def setup_charuco_board():
    """Initialize the ChArUco board with the same parameters used for generation."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), 
                                 SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    # board.setLegacyPattern(True) 
    return board, dictionary

def calibrate_camera(images_folder):
    """Calibrate camera using ChArUco board images."""
    # Initialize the ChArUco board
    board, dictionary = setup_charuco_board()
    
    # Initialize arrays for storing object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    # Get all calibration images
    images = glob.glob(os.path.join(images_folder, "*.jpg")) + \
             glob.glob(os.path.join(images_folder, "*.png"))
    
    if not images:
        print(f"No images found in {images_folder}")
        return None, None
    
    print(f"Found {len(images)} images for calibration")
    
    # Get all possible corners from the board
    all_corners = board.getChessboardCorners()
    print(f"Total possible corners: {len(all_corners)}")
    
    # Process each image
    for fname in images:
        print(f"\nProcessing {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        # Display the grayscale image with detected and rejected markers
        if corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(gray, corners, ids)
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(gray, rejected, None, (0, 0, 255))
        cv2.imshow('Grayscale Image with Markers', gray)
        #cv2.waitKey(500)  # Show for 500ms


        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # Detect ChArUco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            
            if retval > 0 and charuco_ids is not None and len(charuco_ids) > 0:
                # Draw detected ChArUco corners
                cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (255, 0, 0))
                
                # Get the corresponding object points for the detected corners
                obj_corners = all_corners[charuco_ids.flatten()]
                
                # Add points to our arrays
                obj_points.append(obj_corners)
                img_points.append(charuco_corners)
                
                print(f"Detected {len(charuco_corners)} corners in this image")
                
                # Show the image with detected markers and corners
                cv2.imshow('Calibration Image', img)
                cv2.waitKey(500)  # Show each image for 500ms
    
    cv2.destroyAllWindows()
    
    if not obj_points or not img_points:
        print("No valid calibration data found")
        return None, None
    
    # Verify that all point arrays have the same length
    if len(obj_points) != len(img_points):
        print(f"Error: Number of object points ({len(obj_points)}) does not match number of image points ({len(img_points)})")
        return None, None
    
    # Get image size
    img_size = gray.shape[::-1]
    
    # Calibrate camera
    print("\nCalibrating camera...")
    print(f"Number of images used for calibration: {len(obj_points)}")
    
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print(f"Total error: {mean_error/len(obj_points)}")
        print("\nCamera Matrix:")
        print(mtx)
        print("\nDistortion Coefficients:")
        print(dist)
        
        # Save calibration results
        np.savez('camera_calibration.npz', 
                 camera_matrix=mtx, 
                 dist_coeffs=dist, 
                 rvecs=rvecs, 
                 tvecs=tvecs)
        
        return mtx, dist
        
    except cv2.error as e:
        print(f"Calibration failed: {e}")
        return None, None

def main():
    # Create calibration_images folder if it doesn't exist
    if not os.path.exists(CALIBRATION_FOLDER):
        os.makedirs(CALIBRATION_FOLDER)
        print("Created 'calibration' folder. Please add calibration images there.")
        return
    
    # Calibrate camera
    camera_matrix, dist_coeffs = calibrate_camera(CALIBRATION_FOLDER)
    
    if camera_matrix is not None and dist_coeffs is not None:
        print("\nCalibration completed successfully!")
        print("Results saved to 'camera_calibration.npz'")
        print("\nTo use these parameters in your detection script, update the following variables:")
        print("CAMERA_MATRIX = np.array([")
        print(f"    [{camera_matrix[0,0]:.2f}, {camera_matrix[0,1]:.2f}, {camera_matrix[0,2]:.2f}],")
        print(f"    [{camera_matrix[1,0]:.2f}, {camera_matrix[1,1]:.2f}, {camera_matrix[1,2]:.2f}],")
        print(f"    [{camera_matrix[2,0]:.2f}, {camera_matrix[2,1]:.2f}, {camera_matrix[2,2]:.2f}]")
        print("])")
        print("\nDIST_COEFFS = np.array([", end="")
        print(", ".join([f"{x:.6f}" for x in dist_coeffs.flatten()]), end="")
        print("])")

if __name__ == "__main__":
    main() 