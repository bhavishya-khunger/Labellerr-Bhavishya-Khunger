import cv2
import json
import time
import os
from ultralytics import YOLO
import supervision as sv

def process_video(video_path, model_path, num_samples=5):
    """
    Processes a video, yields progress, and saves a few sample frames.
    """
    SAMPLES_DIR = "sample_frames"
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    # Define the output video path
    output_video_path = "annotated_video.mp4"

    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    total_frames = video_info.total_frames
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    json_results = []
    sample_image_paths = []

    sample_interval = total_frames // num_samples if num_samples > 0 else -1
    
    start_time = time.time()

    with sv.VideoSink(output_video_path, video_info) as sink:
        for frame_number, frame in enumerate(sv.get_video_frames_generator(video_path)):
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)

            labels = []
            if tracked_detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {model.model.names[class_id]}"
                    for class_id, tracker_id 
                    in zip(tracked_detections.class_id, tracked_detections.tracker_id)
                ]

            annotated_frame = box_annotator.annotate(frame.copy(), detections=tracked_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=labels)
            
            # Write frame to the video file
            sink.write_frame(annotated_frame)

            if sample_interval > 0 and frame_number % sample_interval == 0 and len(sample_image_paths) < num_samples:
                sample_path = os.path.join(SAMPLES_DIR, f"frame_{frame_number}.jpg")
                cv2.imwrite(sample_path, annotated_frame)
                sample_image_paths.append(sample_path)

            if tracked_detections.tracker_id is not None:
                for bbox, class_id, tracker_id in zip(tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id):
                    json_results.append({
                        "frame_number": frame_number, "tracker_id": int(tracker_id),
                        "class": model.model.names[class_id], "bounding_box": [int(coord) for coord in bbox]
                    })

            elapsed_time = time.time() - start_time
            avg_time_per_frame = elapsed_time / (frame_number + 1)
            remaining_frames = total_frames - (frame_number + 1)
            eta_seconds = int(remaining_frames * avg_time_per_frame)
            progress_percent = int(((frame_number + 1) / total_frames) * 100)
            
            yield {
                "progress": progress_percent, "eta": eta_seconds,
                "annotated_frame": annotated_frame, "is_done": False
            }

    json_output = json.dumps(json_results, indent=4)
    yield {
        "progress": 100, "json_data": json_output,
        "sample_paths": sample_image_paths, 
        "video_path": output_video_path, # <-- ADDED THIS LINE
        "is_done": True
    }