import cv2
import torch
import subprocess
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov10n.pt')  # Use 'yolov8n.pt' or 'yolov8s.pt', depending on your file

# Define the class ID for person (YOLO usually assigns 'person' the class ID 0)
PERSON_CLASS_ID = 0

# Function to detect people in water using YOLOv8 and save the output to a video file
def detect_people_in_water(video_source, output_file="output.mp4"):
    # Capture video from the file or camera
    cap = cv2.VideoCapture(video_source)

    # Check if the video capture is successful
    if not cap.isOpened():
        print("Error opening video file or stream")
        return

    # Get the width, height, and frames per second (fps) of the input video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the video
    temp_output_file = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v for broader compatibility
    out = cv2.VideoWriter(temp_output_file, fourcc, fps, (frame_width, frame_height))

    # Loop through the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        # Break the loop if the video ends
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Loop through the detection results
        for result in results:
            # Filter only the 'person' class detections
            for bbox, class_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                if int(class_id) == PERSON_CLASS_ID and conf > 0.5:
                    # Draw a bounding box around the detected person
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'ALERT DROWNING: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Write the frame with bounding boxes to the temporary output video file
        out.write(frame)

        # Display the frame in real-time
        cv2.imshow('YOLOv8 People Detection', frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Convert the temporary output file to ensure compatibility using FFmpeg
    final_output_file = "drowning_1_processed.mp4"
    
    # Get the absolute paths for both temporary and final output files
    temp_output_file_path = os.path.abspath(temp_output_file)
    final_output_file_path = os.path.abspath(final_output_file)

    ffmpeg_command = [
        'ffmpeg',
        '-i', temp_output_file_path,  # Input file
        '-c:v', 'libx264',            # Use H.264 codec
        '-preset', 'fast',            # Encoding speed
        '-crf', '23',                 # Constant rate factor (quality)
        '-c:a', 'aac',                # Audio codec (if audio is present)
        '-b:a', '192k',               # Audio bitrate (if audio is present)
        '-movflags', '+faststart',    # Enable fast start for web
        final_output_file_path        # Output file
    ]

    # Run FFmpeg command
    subprocess.run(ffmpeg_command, check=True)

    # Clean up temporary files
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    print(f"Output video saved as {final_output_file_path}")

# Example usage with a video file, output will be saved to "drowning_1_processed.mp4"
detect_people_in_water('drowning_1.mp4')  # Replace with your video file
