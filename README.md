# Instance Segmentation and Tracking with YOLO11

This script uses the YOLO11 model for instance segmentation and object tracking. It detects objects, tracks them, and annotates frames with person count and density level (Low/High).
Features

   - Person tracking with unique IDs.
   - Annotated frames with density level and person count.
   - Generates output video with segmentation and tracking.

## Requirements

   - Python 3.8+
   - See requirements.txt for dependencies.

## Installation

   - Clone the repository:

git clone https://github.com/Abdull4h-a/yolo11-tracking.git
cd yolo11-tracking

## Install the dependencies:

    pip install -r requirements.txt

## Usage

Run the script with a video file:

python trace.py <video_path> [output_path]

## Example

python trace.py video.mp4 output.avi

This processes the input video video.mp4 and saves the annotated video as output.avi.