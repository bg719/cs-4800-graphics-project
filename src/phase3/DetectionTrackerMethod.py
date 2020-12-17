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

# Input frame number as input for this method
def detect_in_frame():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("SecondModel\\detection_model-ex-030--loss-0018.300.h5")
    detector.setJsonPath("SecondModel\\detection_config.json")
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image="TestFrames\\frame0.jpg",
        output_image_path="TestFrames\\frame0_detected.jpg",
        minimum_percentage_probability=30,
        display_object_name=False,
        display_percentage_probability=False)


    return detections

def detect_in_frame_test(vid_frame):
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("SecondModel\\detection_model-ex-030--loss-0018.300.h5")
    detector.setJsonPath("SecondModel\\detection_config.json")
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=vid_frame, input_type="array",
        output_type="array",
        minimum_percentage_probability=30,
        display_object_name=False,
        display_percentage_probability=False)

    return detections


def coord_math(box_list):
    bboxes = []
    colors = []


    for detection in box_list[1]:
        box_pts = detection["box_points"]
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)
        # Creates different colors for the boxes randomly
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    return bboxes, colors


def updated_rectangles(vid_frame, found_boxes, colors):

    for i, newbox in enumerate(found_boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        cv2.rectangle(vid_frame, p1, p2, colors[i], 2, 1)

    return frame

def get_multitracker():

    return cv2.MultiTracker_create()



if __name__ == "__main__":
    vid_name = 'mygeneratedvideo.mp4'
    # Counter for frames
    count = 0
    cap = cv2.VideoCapture(vid_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    vid_write = result = cv2.VideoWriter('filename.avi',
                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                         20, size)
    trackerName = "CSRT"
    multiTracker = None
    cord = None
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if count % 10 == 0: #count == 0:
            multiTracker = get_multitracker()
            dif = detect_in_frame_test(frame)
            cord = coord_math(dif)
            for bbox in cord[0]:
                multiTracker.add(createTrackerByName(trackerName), frame, tuple(bbox))

            mu = updated_rectangles(frame, cord[0], cord[1])
            vid_write.write(mu)

        else:

            success, boxes = multiTracker.update(frame)
            mu = updated_rectangles(frame, boxes, cord[1])
            vid_write.write(mu)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        count += 1
    print(cord[0])
