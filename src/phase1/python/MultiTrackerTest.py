# Colin Henson and Bryce George
# CS 4800 Computer Graphics
# Tests different tracking algorithms using OpenCV MultiTracker

from __future__ import print_function
import sys
import cv2
from random import randint
import time


# Test different trackers
def createTrackerByName(trackerType):
    Tracker = None
    # Kernelized Correlation Filter
    if trackerType == "KCF":
        Tracker = cv2.TrackerKCF_create()
    # General Object using Regression Networks
    elif trackerType == "GOTURN":
        Tracker = cv2.TrackerGOTURN_create()
    # Discriminative Correlation Filter Tracker with Channel and Spacial Reliability
    elif trackerType == "CSRT":
        Tracker = cv2.TrackerCSRT_create()
    # Defaults to None
    else:
        Tracker = None

    return Tracker


# Name of video/Path of video
videoPath = "video.avi"

# Opens the video to be read
cap = cv2.VideoCapture(videoPath)

# Reads the video into memory
success, frame = cap.read()

# Checks to see if it can read video
if not success:
    print('Failed to read video')
    sys.exit(1)
# Initialize lists for bounding boxes and random colors
bboxes = []
colors = []

# User selects region of interest
# 1. Select target object with mouse
# 2. Press space
# 3. To select more boxes, press space again
# 4. When done, press q
while True:
    # Creates the object selection window
    bbox = cv2.selectROI("MultiTracker", frame)
    # Adds box coordinates to list
    bboxes.append(bbox)
    # Picks random colors for bounding boxes
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select new object")
    # Waits until user presses q
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break

# Purely for testing purposes
# Displays coordinates of boxes in console
print('Selected Bounding Boxes {}'.format(bboxes))
# Defines which tracking algorithm you want to use
# Recommend CSRT, works best in testing
trackerType = "CSRT"
# Creates the multi tracker object
multiTracker = cv2.MultiTracker_create()

# Iterates through bounding box list
for bbox in bboxes:
    # Adds a tracker for each bounding box
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
# While loop to tracks objects in video
while cap.isOpened():
    # Makes sure video has been read
    success, frame = cap.read()
    if not success:

        break
    # Time delay between frame rendering
    # Without time delay, frames are rendered as fast as processed
    time.sleep(.1)
    # Update tracking on video frame
    success, boxes = multiTracker.update(frame)
    # Creates a new box around the tracked element for each frame
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    # Shows next tracked frame
    cv2.imshow('MultiTracker', frame)

    # Waits until video is done or user pressed Esc
    if cv2.waitKey(1) & 0xFF == 27: 
        break


