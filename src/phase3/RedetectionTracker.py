from imageai.Detection.Custom import CustomObjectDetection
import cv2
from random import randint
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def create_tracker(tracker_type):
    tracker = None
    # Kernelized Correlation Filter
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    # General Object using Regression Networks
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    # Discriminative Correlation Filter tracker with Channel and Spacial Reliability
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    # Defaults to None
    else:
        tracker = None

    return tracker


def detect_in_frame(vid_frame, model_path, config_path):
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(config_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=vid_frame, input_type="array",
        output_type="array",
        minimum_percentage_probability=30,
        display_object_name=False,
        display_percentage_probability=False)

    return detections


def convert_box_coords(box_list):
    bboxes = []
    for detection in box_list[1]:
        box_pts = detection["box_points"]
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)

    return bboxes


def create_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def draw_bounding_boxes(vid_frame, found_boxes, colors):

    for i, newbox in enumerate(found_boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        cv2.rectangle(vid_frame, p1, p2, colors[i], 2, 1)

    return frame

def get_multitracker():

    return cv2.MultiTracker_create()

def calc_detected_centroids(detections):
    centroids = []

    for detection in detections[1]:
        box_points = detection["box_points"]
        centerCord = ((box_points[0] + box_points[2]) / 2, (box_points[1] + box_points[3]) / 2)
        centroids.append(centerCord)

    return centroids

def calc_tracked_centroids(detections):
    centroids = []

    for box_points in detections:

        centerCord = (box_points[0] + (box_points[2]/2), box_points[1] + (box_points[3]/2))
        centroids.append(centerCord)

    return centroids

def find_nearest(detected_centroids, tracked_centroids):
    detected_points = np.array(detected_centroids)
    tracked_points = np.array(tracked_centroids)

    nbrs = NearestNeighbors(n_neighbors=len(detected_points), algorithm='ball_tree').fit(detected_points)
    distances, indices = nbrs.kneighbors(tracked_points)
    return distances, indices

if __name__ == "__main__":
    vid_name = 'mygeneratedvideo.mp4'
    model_path = "SecondModel\\detection_model-ex-030--loss-0018.300.h5"
    config_path = "SecondModel\\detection_config.json"

    # Counter for frames
    count = 0
    detection_interval = 10
    margin_scalar = 0.5

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
    trackerType = "CSRT"
    multiTracker = None
    coords = None
    colors = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if count == 0:
            multiTracker = get_multitracker()
            dif = detect_in_frame(frame, model_path, config_path)
            coords = convert_box_coords(dif)
            for bbox in coords:
                colors.append(create_color())
                multiTracker.add(create_tracker(trackerType), frame, tuple(bbox))

            mu = draw_bounding_boxes(frame, coords, colors)
            vid_write.write(mu)

        elif count % detection_interval == 0:
            success, boxes = multiTracker.update(frame)

            # Save colors
            old_colors = colors.copy()
            colors.clear()

            # Set the comparison threshold
            index_smallest = max(boxes, key=lambda t: t[2])
            threshold = index_smallest[2] * margin_scalar

            # Reinitialize multi-tracker
            multiTracker = get_multitracker()

            # Run re-detection on current frame
            dif = detect_in_frame(frame, model_path, config_path)

            # Calculate centroids for detected and tracked bounding boxes
            detected_centroids = calc_detected_centroids(dif)
            tracked_centroids = calc_tracked_centroids(boxes)

            # Find the nearest neighbor in the detected set to the tracked set
            nearest = find_nearest(detected_centroids, tracked_centroids)

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

            coords = convert_box_coords(dif)

            while len(colors) < len(coords):
                colors.append(create_color())

            for bbox in coords:
                multiTracker.add(create_tracker(trackerType), frame, tuple(bbox))

            print(len(colors))
            print(len(coords))
            mu = draw_bounding_boxes(frame, coords, colors)
            vid_write.write(mu)

        else:
            success, boxes = multiTracker.update(frame)
            mu = draw_bounding_boxes(frame, boxes, colors)
            vid_write.write(mu)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("Frame " + str(count) + " done.")
        count += 1
    #print(cord[0])
