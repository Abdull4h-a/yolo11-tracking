import sys
import cv2
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def main(video_path, output_path="output.avi"):
    # Initialize tracking history
    track_history = defaultdict(lambda: [])
    
    # Load YOLO model
    model = YOLO("yolo11n-seg.pt")  # Ensure the model is in the same directory or provide the correct path
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)
    
    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    if fps < 1:
        print("Warning: Invalid FPS detected, setting default FPS to 30.")
        fps = 30
    
    # Setup video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not out.isOpened():
        print(f"Error: Cannot open output file {output_path}")
        sys.exit(1)
    
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video processing completed.")
            break
        
        annotator = Annotator(im0, line_width=2)
        results = model.track(im0, persist=True)
        person_count = 0

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            person_count = len(track_ids)
            
            for mask, track_id in zip(masks, track_ids):
                color = colors(int(track_id), True)
                txt_color = annotator.get_txt_color(color)
                annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)

        # Calculate density level
        density_level = "Low" if person_count < 10 else "High"
        
        # Create overlay text
        line1 = f"Person Count: {person_count}"
        line2 = f"Density Level: {density_level}"

        # Draw a clean, compact overlay box
        box_x, box_y, box_w, box_h = 10, 10, 300, 80  # Dimensions of overlay box
        overlay_color = (0, 0, 0)  # Black box
        overlay_alpha = 0.6  # Transparency level
        overlay = im0.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), overlay_color, -1)  # Draw rectangle
        cv2.addWeighted(overlay, overlay_alpha, im0, 1 - overlay_alpha, 0, im0)  # Apply transparency

        # Add text lines inside the overlay box
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        cv2.putText(im0, line1, (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        cv2.putText(im0, line2, (box_x + 10, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Write to output video and display frame
        out.write(im0)
        cv2.imshow("Instance Segmentation and Tracking", im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trace.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.avi"
    main(video_path, output_path)
