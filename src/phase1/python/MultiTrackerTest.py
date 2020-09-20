from __future__ import print_function
import sys
import cv2
from random import randint
import time

# Test different trackers
def createTrackerByName(trackerType):
    Tracker = None
    if trackerType == "KCF":
        Tracker = cv2.TrackerKCF_create()
    elif trackerType == "GOTurn":
        Tracker = cv2.TrackerGOTURN_create()
    elif trackerType == "CSRT":
        Tracker = cv2.TrackerCSRT_create()
    else:
        Tracker = None

    return Tracker
# Name of video
videoPath = "secondtestvideo.avi"
# Video capture video
cap = cv2.VideoCapture(videoPath)

success, frame = cap.read()
# Checks to see if it can read video
if not success:
    print('Failed to read video')
    sys.exit(1)
# Initialize lists
bboxes = []
colors = []
# ROI while loop to find bounded boxes
while True:
    bbox = cv2.selectROI("MultiTracker", frame)
    bboxes.append(bbox)
    # Make boxes random colors
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select new object")
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break

print('Selected Bounding Boxes {}'.format(bboxes))
# Use CSRT, works best
trackerType = "CSRT"
multiTracker = cv2.MultiTracker_create()

for bbox in bboxes:
    # Create multitracker using tracking algorithm above
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
# While loop to tracks objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:

        break
        # Time delay
    time.sleep(.25)
    # Update tracking on frame
    success, boxes = multiTracker.update(frame)
    # Creates a new box around the tracked element
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    # Shows next tracked frame
    cv2.imshow('MultiTracker', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


