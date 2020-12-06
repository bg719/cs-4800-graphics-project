from imageai.Detection.Custom import CustomObjectDetection
import cv2
from random import randint
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


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



    for detection in box_list[1]:
        box_pts = detection["box_points"]
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)
        # Creates different colors for the boxes randomly


    return bboxes


def create_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def updated_rectangles(vid_frame, found_boxes, colors):

    for i, newbox in enumerate(found_boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        cv2.rectangle(vid_frame, p1, p2, colors[i], 2, 1)

    return frame

def get_multitracker():

    return cv2.MultiTracker_create()

def calcDetectedCentroids(detections):
    centroids = []

    for detection in detections[1]:
        box_points = detection["box_points"]
        centerCord = ((box_points[0] + box_points[2]) / 2, (box_points[1] + box_points[3]) / 2)
        centroids.append(centerCord)

    return centroids

def calcTrackedCentroids(detections):
    centroids = []

    for box_points in detections:

        centerCord = (box_points[0] + (box_points[2]/2), box_points[1] + (box_points[3]/2))
        centroids.append(centerCord)

    return centroids

def findNearest(detected_centroids, tracked_centroids):
    detected_points = np.array(detected_centroids)
    tracked_points = np.array(tracked_centroids)

    nbrs = NearestNeighbors(n_neighbors=len(detected_points), algorithm='ball_tree').fit(detected_points)
    distances, indices = nbrs.kneighbors(tracked_points)
    return distances, indices

if __name__ == "__main__":
    vid_name = 'mygeneratedvideo.mp4'
    # Counter for frames
    count = 0
    cap = cv2.VideoCapture(vid_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    dt = time.strftime("%m-%d-%y  %I %M %S %p")
    extension = "avi"
    file_name = dt + extension

    vid_write = result = cv2.VideoWriter(file_name,
                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                         20, size)
    trackerName = "CSRT"
    multiTracker = None
    cord = None
    colors = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if count == 0:

            multiTracker = get_multitracker()
            dif = detect_in_frame_test(frame)
            cord = coord_math(dif)
            for bbox in cord:
                colors.append(create_color())
                multiTracker.add(createTrackerByName(trackerName), frame, tuple(bbox))

            mu = updated_rectangles(frame, cord, colors)
            vid_write.write(mu)

        elif count % 10 == 0: #count == 0:
            success, boxes = multiTracker.update(frame)

            old_colors = colors.copy()
            colors.clear()

            print(old_colors)
            index_smallest = max(boxes, key=lambda t: t[2])
            threshold = index_smallest[2] * 0.5

            multiTracker = get_multitracker()
            dif = detect_in_frame_test(frame)
            calcDetected = calcDetectedCentroids(dif)
            calcTracked = calcTrackedCentroids(boxes)
            nearest = findNearest(calcDetected, calcTracked)

            for i in range(len(nearest[0])):
                print(nearest[0][i][0])
                if nearest[0][i][0] <= threshold and i < len(old_colors):
                    print("Found existing bowl")
                    colors.append(old_colors[i])
                else:
                    colors.append(create_color())

            #print(nearest)
            # im = plt.imread(frame)
            # implot = plt.imshow(frame)
            # for i in range(len(calcDetected)):
            #     plt.scatter(calcDetected[i][0], calcDetected[i][1], c='r')
            # for i in range(len(calcTracked)):
            #     plt.scatter(calcTracked[i][0], calcTracked[i][1], c='b')
            # plt.show()

            cord = coord_math(dif)

            while len(colors) < len(cord):
                colors.append(create_color())

            for bbox in cord:
                multiTracker.add(createTrackerByName(trackerName), frame, tuple(bbox))

            print(len(colors))
            print(len(cord))
            mu = updated_rectangles(frame, cord, colors)
            vid_write.write(mu)

        else:

            success, boxes = multiTracker.update(frame)
            mu = updated_rectangles(frame, boxes, colors)
            vid_write.write(mu)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("Frame " + str(count) + " done.")
        count += 1
    #print(cord[0])
