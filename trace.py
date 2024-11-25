import sys
import cv2
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def main(video_path, output_path="output.avi"):
    # Initialize tracking history
    track_history = defaultdict(lambda: [])
    
    # Load YOLO model
    model = YOLO("yolo11n-seg.pt")  # Make sure the model is in the same directory or provide the path
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    
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

        density_level = "Low" if person_count < 10 else "High"
        text = f"Person Count: {person_count} | Density Level: {density_level}"
        cv2.putText(im0, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
