from constants import DATA_DIR, VIDEOS_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, WIDTH, HEIGHT, CLASSES

import pathlib
import json
import cv2
from tqdm import tqdm


def process_and_resize_video(video_info, output_path):
    """
    Verarbeitet ein Video, indem der Teil der Bounding Box ausgeschnitten und dieser dann skaliert wird.
    Auch wird das Video nach angegebenen Start- und Endframe zugeschnitten.
    
    Args:
        video_info: Video Metadaten, beinhaltet Pfad, Bounding Box, Start- und Endframe.
        output_path: Pfad wo verarbeitete Videos gespeichert werden.
    
    Returns:
        bool: True wenn Verarbeitung erfolgreich, ansonsten False
    """
    try:
        cap = cv2.VideoCapture(str(video_info['path']))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_info['path']}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = video_info.get('fps', original_fps)
        
        # Start-/Endframe ermitteln
        frame_start = max(0, video_info['frame_start'] - 1)
        frame_end = video_info['frame_end'] if video_info['frame_end'] != -1 else total_frames
        frame_end = min(frame_end, total_frames)
        
        # Bounding Box ermitteln
        bbox = video_info['bbox']
        x_min, y_min, x_max, y_max = bbox
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (WIDTH, HEIGHT))
        
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count >= frame_start and frame_count < frame_end:
                # Frame zu Bounding Box schneiden
                h, w = frame.shape[:2]
                
                x_min_clipped = max(0, min(x_min, w))
                y_min_clipped = max(0, min(y_min, h))
                x_max_clipped = max(0, min(x_max, w))
                y_max_clipped = max(0, min(y_max, h))
                
                if x_max_clipped <= x_min_clipped or y_max_clipped <= y_min_clipped:
                    print(f"Warning: Invalid bounding box for {video_info['path']}")
                    frame_count += 1
                    continue
                
                cropped_frame = frame[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]
                resized_frame = cv2.resize(cropped_frame, (WIDTH, HEIGHT))
                
                out.write(resized_frame)
                processed_frames += 1
            
            frame_count += 1
            
            # Stop wenn alle zu verarbeitenden Frames verarbeitet wurden
            if frame_count >= frame_end:
                break
        
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

def validate_entry(entry):
    splits = []
    for inst in entry["instances"]:
        path = pathlib.Path(VIDEOS_DIR / f"{inst["video_id"]}.mp4")
        if path.exists():
            splits.append(inst["split"])
    
    required_splits = {"train", "test", "val"}
    return required_splits.issubset(set(splits))


def main():
    with open('./data/WLASL_v0.3.json', 'r') as file:
        data = json.load(file)
    
    class_count = 0
    videos = {}
    for entry in data:
        if (CLASSES == -1 or class_count < CLASSES) and validate_entry(entry):
            for inst in entry["instances"]:
                path = pathlib.Path(VIDEOS_DIR / f"{inst["video_id"]}.mp4")
                if path.exists():
                    videos[inst["video_id"]] = {
                    "path": path,
                    "split": inst["split"],
                    "gloss": entry["gloss"],
                    "bbox": inst["bbox"],
                    "fps": inst["fps"],
                    "frame_start": inst["frame_start"],
                    "frame_end": inst["frame_end"],
                    }
                
            class_count += 1
    
    successful = 0
    failed = 0

    for video_id, video_info in tqdm(videos.items(), desc="Processing videos"):
        path = DATA_DIR / f"{video_info["split"]}/{video_info["gloss"]}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Dateipfad entsprechend der im Modell erwarteten Namensstruktur erstellen
        output_filename = f"{video_info["gloss"]}_{video_info["split"]}_{video_id}.mp4"
        output_path = path / output_filename
        
        # Überspringe wenn Datei bereits existiert, kein unnötiges Verarbeiten von Videos
        if output_path.exists():
            print(f"Skipping {output_filename} - already exists")
            continue
        
        if process_and_resize_video(video_info, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Classes processed: {class_count}")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed to process: {failed} videos")


if __name__ == "__main__":
    main()

