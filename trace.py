import sys
import cv2
import numpy as np
from ultralytics import YOLO

def draw_info(frame, count):
    """
    Draw an overlay on the video frame displaying person count and density level.
    """
    # Create overlay
    overlay = np.zeros((100, 320, 3), dtype=np.uint8)
    overlay.fill(30)  # Dark gray background

    # Determine density level and color
    density_level = "High" if count >= 10 else "Low"
    density_color = (0, 0, 255) if count >= 10 else (0, 255, 0)  # Red for High, Green for Low

    # Add text to overlay
    cv2.putText(overlay, f"Person Count: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, f"Density Level: {density_level}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, density_color, 2)

    # Blend overlay with frame
    frame[10:110, 10:330] = cv2.addWeighted(frame[10:110, 10:330], 0.2, overlay, 0.8, 0)

    return frame

def main(video_path, output_path="output.avi"):
    """
    Process a video to detect and track people using YOLO, and display count/density information.
    """
    # Load YOLO model
    model = YOLO("yolo11l-seg.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 30)

    # Initialize video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    print("Processing video... Press 'q' to quit or 'p' to pause.")
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video processing complete.")
                break

            # Perform YOLO detection
            results = model.track(frame, persist=True, classes=[0])  # Class 0 for 'person'
            person_count = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                person_count = len(boxes)

                # Draw bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Add overlay with person count and density level
            frame = draw_info(frame, person_count)

            # Write frame to output
            out.write(frame)

            # Show frame
            cv2.imshow("Person Detection", frame)
            print(f"Processing frame {frame_count}...", end="\r")
            frame_count += 1

            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Quit
                print("\nExiting...")
                break
            elif key == ord("p"):  # Pause
                print("\nPaused. Press any key to continue.")
                cv2.waitKey(0)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.avi"
    main(video_path, output_path)
