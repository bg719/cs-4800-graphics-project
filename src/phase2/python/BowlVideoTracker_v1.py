from imageai.Detection.Custom import CustomObjectDetection
import os
from os.path import join
import cv2


# Currying method (https://www.python-course.eu/currying_in_python.php)
def compose(g, f):
    def h(*args, **kwargs):
        return g(f(*args, **kwargs))
    return h


# Create an object tracker by type
def createTracker(tracker_type='CSRT'):
    tracker = None
    # Kernelized Correlation Filter
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    # General Object Tracking Using Regression Networks
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    # Discriminative Correlation Filter Tracker with Channel and Spatial Reliability
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()

    # Return the tracker
    return tracker


def trackBowls(second, detected_objects, detected_object_counts, average_detected_object_count, detected_frame):
    multiTracker = cv2.MultiTracker_create()

    for bowl in detected_objects:
        multiTracker.add(createTracker(), detected_frame, bowl['box_points'])


    return


if __name__ == "__main__":
    execution_path = os.getcwd()
    input_file = 'video.avi'
    output_file = ''

    video_detector = CustomObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(join(execution_path, "yolo.h5"))
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(input_file_path=join(execution_path, input_file),
                                          output_file_path=join(execution_path, output_file),
                                          frames_per_second=25,
                                          frames_per_second_function=trackBowls,
                                          minimum_percentage_probability=30)


# Process:
# 1) Run bowl detection on first frame
# 2) Start CSRT using the bboxes detected from step 1
# 3) Enter loop and run CSRT until: current_frame % track_for_frames == 0
# 4) Re-run bowl detection on last frame tracked
# 5) Compare bboxes from tracker to bboxes from object detection on the same frame (i.e.
#    any frame where current_frame % track_for_frames == 0)
# 6) Assume boxes with closest centroids between the two sets are the same object
# 7) Add objects to new MultiTracker, providing the same id/color to objects which were deemed
#    to be the same between the previous tracking segment and the most current re-detection.
#    At this point, remove any objects which have gone out of frame and add any new objects which
#    have come into frame. (Prefer detection over tracking?)
# 8) Cease outputting bowl paths and bounds from old MultiTracker and begin using new MultiTracker
# 9) Repeat loop until the video source is closed.
