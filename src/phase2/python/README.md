# Phase 2 Python Code

A repository for Python code developed in Phase 2.

### Files:

* `DetectionTrackerPrototype.py` : A script testing the integration of the ImageAI YOLO_V3 object detection with the OpenCV CSRT trackers aggregated within 
a  Multi-Tracker object. This prototype runs object detection on the first frame of the video and passes the bounding boxes of the discovered objects to the 
OpenCV trackers to monitor for the remainder of the video. 

    ![Prototype output](../../demos/prototype.gif)

* `ModelEvaluation.py` : A script for evaluating YOLO_V3 object models generated using ImageAI.

* `ModelTrainer.py` : A script forr training a custom YOLO_V3 object detector with an annotated image data set using ImageAI.

* `SingleFrameDetection.py` : A script which applies a custom YOLO_V3 object detector on a single video frame.

* `VideoDetection.py` : A script which applies a custom YOLO_V3 object dector to a video file. 
