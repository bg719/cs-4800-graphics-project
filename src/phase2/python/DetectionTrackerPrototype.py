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
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("detection_model-ex-015--loss-0015.368.h5")
    detector.setJsonPath("detection_config.json")
    detector.loadModel()

    vidcap = cv2.VideoCapture('mygeneratedvideo.mp4')
    success, frames = vidcap.read()
    # Counter
    count = 0
    os.mkdir("TestFrames")
    # While there are frames in the video
    while success:
        # Write frame with name of number
        cv2.imwrite("TestFrames\\frame%d.jpg" % count, frames)
        # Checks for next frame
        success, image = vidcap.read()
        # print('Read a n ew frame: ', success)
        count += 1

    time.sleep(2)
    trackerName = "CSRT"
    multiTracker = cv2.MultiTracker_create()
    bboxes = []
    colors = []
    detections = detector.detectObjectsFromImage(input_image="TestFrames\\frame0.jpg",
                                                 output_image_path="TestFrames\\frame0_detected.jpg",
                                                 display_object_name=False, display_percentage_probability=False)

    for detection in detections:
        box_pts = detection["box_points"]
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    print(bboxes)
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerName), frames, tuple(bbox))

    cap = cv2.VideoCapture('mygeneratedvideo.mp4')

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        time.sleep(.1)
        success, boxes = multiTracker.update(frame)

        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        cv2.imshow("MultiTracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    #shutil.rmtree("TestFrames")





