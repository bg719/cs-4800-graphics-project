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

import os
from os.path import join
import cv2


# Create an object tracker by type
def create_tracker(tracker_type='CSRT'):
    """
    Creates a new object tracker.

    :param tracker_type: the type of the tracker to be created: "KCF", "GOTURN", or "CSRT" (The default is CSRT)
    :return: the object tracker
    """
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

def frame_generator(video_src='video.avi'):
    """
    Extracts the frames from the video source and yield returns them one at a time.

    :param video_src: the video source file
    :returns: the frame number and the image
    """
    # create new video capture
    capture = cv2.VideoCapture(video_src)

    # check readability of video
    success, image = capture.read()

    #initialize counter
    count = 0

    while success:
        # yield the frame and its number
        yield count, image

        # get the next frame from the source
        success, image = capture.read()
        count += 1

    yield -1, None


def track_video(video_src='video.avi'):
    return


if __name__ == "__main__":
    exit()
