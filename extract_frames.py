import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video and save them as images.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder where frames will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame as image
        output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Finished! Extracted {frame_count} frames to {output_folder}")

def main():
    # parser = argparse.ArgumentParser(description='Extract frames from a video file')
    # parser.add_argument('video_path', help='Path to the input video file')
    # parser.add_argument('output_folder', help='Path to the output folder for frames')
    
    # args = parser.parse_args()
    video_path = "data/clean/VID20250521123919.mp4"
    output_folder = "calibration_images"
    extract_frames(video_path, output_folder)

if __name__ == "__main__":
    main() 