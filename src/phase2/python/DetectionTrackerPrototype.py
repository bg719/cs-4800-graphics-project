# Colin Henson and Bryce George Bowl Project
# For Demo 2
# 11/3/2020

from imageai.Detection.Custom import CustomObjectDetection
import cv2
import os
import time
from random import randint
import shutil

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


if __name__ == "__main__":

    # Set up the ImageAI Detection
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("SecondModel\\detection_model-ex-030--loss-0018.300.h5")
    detector.setJsonPath("SecondModel\\detection_config.json")
    detector.loadModel()

    # Setup video read and write
    vidcap = cv2.VideoCapture('ThreePoolTestVid.avi')
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    size = (frame_width, frame_height)
    vid_write = result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)
    success, frames = vidcap.read()
    # Counter for frames
    count = 0
    # Folder for frames
    os.mkdir("TestFrames")
    # While there are frames in the video
    while success:
        # Write frame with name of number
        cv2.imwrite("TestFrames\\frame%d.jpg" % count, frames)
        # Checks for next frame
        success, image = vidcap.read()
        # print('Read a n ew frame: ', success)
        count += 1
    # Sleeps to make sure folder is filled
    time.sleep(2)
    # Creates multitracker type
    trackerName = "CSRT"
    multiTracker = cv2.MultiTracker_create()
    # Lists for storing box info
    bboxes = []
    colors = []
    # Detects bowls in the first frame of the video
    detections = detector.detectObjectsFromImage(input_image="TestFrames\\frame0.jpg",
                                                 output_image_path="TestFrames\\frame0_detected.jpg",
                                                 minimum_percentage_probability=30,
                                                 display_object_name=False, display_percentage_probability=False)

    # Gets detection bowls, translates coordinates and adds them to lists
    for detection in detections:
        box_pts = detection["box_points"]
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)
        # Creates different colors for the boxes randomly
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    print(bboxes)
    # Create trackers for each bowl
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerName), frames, tuple(bbox))
    # Reopens the video
    cap = cv2.VideoCapture('ThreePoolTestVid.avi')

    # While the video has frames
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        time.sleep(.1)
        # Updates multitracker each frame
        success, boxes = multiTracker.update(frame)

        # Plots new box on the image
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        # Writes each frame to a video file
        vid_write.write(frame)
        # cv2.imshow("MultiTracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Deletes test frame folder
    shutil.rmtree("TestFrames")





