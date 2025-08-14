from constants import VIDEOS_DIR, PROCESSED_VIDEOS_DIR, WIDTH, HEIGHT, CLASSES

import pathlib
import json
import cv2
from tqdm import tqdm


def process_and_resize_video(video_info, output_path):
    """
    Process a single video by extracting frames, cropping to bounding box, and resizing.
    
    Args:
        video_info (dict): Video metadata including path, bbox, frame_start, frame_end
        output_path (pathlib.Path): Path where processed video will be saved
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(str(video_info['path']))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_info['path']}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use provided fps if available, otherwise use original
        fps = video_info.get('fps', original_fps)
        
        # Determine frame range
        frame_start = max(0, video_info['frame_start'] - 1)  # Convert to 0-based indexing
        frame_end = video_info['frame_end'] if video_info['frame_end'] != -1 else total_frames
        frame_end = min(frame_end, total_frames)
        
        # Extract bounding box coordinates
        bbox = video_info['bbox']
        x_min, y_min, x_max, y_max = bbox
        
        # Set up video writer with target dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (WIDTH, HEIGHT))
        
        # Process frames
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if current frame is in the desired range
            if frame_count >= frame_start and frame_count < frame_end:
                # Crop frame to bounding box
                h, w = frame.shape[:2]
                
                # Ensure bounding box is within frame dimensions
                x_min_clipped = max(0, min(x_min, w))
                y_min_clipped = max(0, min(y_min, h))
                x_max_clipped = max(0, min(x_max, w))
                y_max_clipped = max(0, min(y_max, h))
                
                # Skip if bounding box is invalid
                if x_max_clipped <= x_min_clipped or y_max_clipped <= y_min_clipped:
                    print(f"Warning: Invalid bounding box for {video_info['path']}")
                    frame_count += 1
                    continue
                
                cropped_frame = frame[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]
                
                # Resize the cropped frame to target dimensions
                resized_frame = cv2.resize(cropped_frame, (WIDTH, HEIGHT))
                
                # Write the processed frame
                out.write(resized_frame)
                processed_frames += 1
            
            frame_count += 1
            
            # Break if we've processed all needed frames
            if frame_count >= frame_end:
                break
        
        # Release resources
        cap.release()
        out.release()
        
        if processed_frames == 0:
            print(f"Warning: No frames processed for {video_info['path']}")
            return False
        
        print(f"Processed and resized {processed_frames} frames for {output_path.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {video_info['path']}: {str(e)}")
        return False


def main():
    with open('./data/WLASL_v0.3.json', 'r') as file:
        data = json.load(file)
    
    videos = {}
    for entry in data:
        if entry["gloss"] in CLASSES:
            for inst in entry["instances"]:
                path = pathlib.Path(VIDEOS_DIR / f"{inst["video_id"]}.mp4")
                if path.exists():
                    videos[inst["video_id"]] = {
                    "path": path,
                    "bbox": inst["bbox"],
                    "fps": inst["fps"],
                    "frame_start": inst["frame_start"],
                    "frame_end": inst["frame_end"],
                    }

    print(f"Found {len(videos)} videos to process")
    
    # Process each video
    successful = 0
    failed = 0

    for video_id, video_info in tqdm(videos.items(), desc="Processing videos"):
        # Create output filename identical to original
        output_filename = f"{video_id}.mp4"
        output_path = PROCESSED_VIDEOS_DIR / output_filename
        
        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {output_filename} - already exists")
            continue
        
        # Process the video
        if process_and_resize_video(video_info, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed to process: {failed} videos")
    print(f"Processed videos saved to: {PROCESSED_VIDEOS_DIR}")


if __name__ == "__main__":
    main()

