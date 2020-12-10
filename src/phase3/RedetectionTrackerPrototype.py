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

    if tracker_type == "KCF":  # Kernelized Correlation Filter
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "GOTURN":  # General Object using Regression Networks
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "CSRT":  # Discriminative Correlation Filter tracker with Channel and Spacial Reliability
        tracker = cv2.TrackerCSRT_create()
    else:  # Defaults to None
        tracker = None

    return tracker


def detect_in_frame(vid_frame, model_path, config_path):
    # Setup the detector
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(config_path)
    detector.loadModel()

    # Detect objects in the frame
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
        # Get bounding box points
        box_pts = detection["box_points"]

        # Calculate the x and y differences
        xdiff = (int(box_pts[2] - int(box_pts[0])))
        ydiff = (int(box_pts[3]) - int(box_pts[1]))

        # Add the converted coordinates
        finished_pnts = (box_pts[0], box_pts[1], xdiff, ydiff)
        bboxes.append(finished_pnts)

    return bboxes


def create_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


def draw_bounding_boxes(vid_frame, found_boxes, colors):
    for i, newbox in enumerate(found_boxes):
        # Set rectangle points
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        # Draw the rectangle
        cv2.rectangle(vid_frame, p1, p2, colors[i], 2, 1)

    return frame


def get_multitracker():
    return cv2.MultiTracker_create()


def calc_detected_centroids(detections):
    centroids = []

    for detection in detections[1]:
        # Get bounding box points
        box_points = detection["box_points"]

        # Calculate the centroid
        centerCord = ((box_points[0] + box_points[2]) / 2, (box_points[1] + box_points[3]) / 2)

        # Add the centroid
        centroids.append(centerCord)

    return centroids


def calc_tracked_centroids(detections):
    centroids = []

    for box_points in detections:
        # Calculate the centroid
        centerCord = (box_points[0] + (box_points[2]/2), box_points[1] + (box_points[3]/2))

        # Add the centroid
        centroids.append(centerCord)

    return centroids


def find_nearest(detected_centroids, tracked_centroids):
    # Convert input lists to arrays
    detected_points = np.array(detected_centroids)
    tracked_points = np.array(tracked_centroids)

    # Fit a nearest neighbors model with the detected points
    nbrs = NearestNeighbors(n_neighbors=len(detected_points), algorithm='ball_tree').fit(detected_points)

    # Calculate the nearest neighbors in the tracked set
    distances, indices = nbrs.kneighbors(tracked_points)

    return distances, indices


if __name__ == "__main__":
    # Video file to be analyzed
    vid_name = 'mygeneratedvideo.mp4'

    # Path to the bowl detection model
    model_path = "SecondModel\\detection_model-ex-030--loss-0018.300.h5"

    # Path to the bowl detection model config
    config_path = "SecondModel\\detection_config.json"

    # Counter for frames
    count = 0

    # The number of frames to track before re-detection
    detection_interval = 10

    # The multiplicative scalar applied to the size-based detection-tracking correlation threshold
    threshold_scalar = 10

    # The tracker type used during tracking sequences
    trackerType = "CSRT"

    # Setup the video stream and capture video info
    cap = cv2.VideoCapture(vid_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    # Generate the output file name
    dt = time.strftime("%m-%d-%y  %I %M %S %p")
    extension = ".avi"
    file_name = dt + extension

    # Create the video writer
    vid_write = result = cv2.VideoWriter(file_name,
                                         cv2.VideoWriter_fourcc(*'MJPG'),
                                         20, size)

    # Initialize local vars
    multiTracker = None
    coords = None
    colors = []

    # Process the video
    while cap.isOpened():
        # Get the next frame
        success, frame = cap.read()
        if not success:
            break

        # If this is the first frame, run the initial detection
        if count == 0:
            print("Detecting objects in frame" + str(frame) + "...")

            # Initialize the multi-tracker
            multiTracker = get_multitracker()

            # Run detection on the frame
            dif = detect_in_frame(frame, model_path, config_path)

            # Convert the bounding box coordinates
            coords = convert_box_coords(dif)

            # Add a tracker for each object detected and assign it a color
            for bbox in coords:
                colors.append(create_color())
                multiTracker.add(create_tracker(trackerType), frame, tuple(bbox))

            # Draw the bounding boxes on the frame
            mu = draw_bounding_boxes(frame, coords, colors)

            # Write the frame
            vid_write.write(mu)

        # If we have reached a re-detection interval, perform the redetection-tracking merge sequence
        elif count % detection_interval == 0:
            print("Re-detecting objects in frame" + str(frame) + "...")

            # Update the positions of all tracked objects
            success, boxes = multiTracker.update(frame)

            # Save colors
            old_colors = colors.copy()
            colors.clear()

            # Set the comparison threshold
            largest_obj_index = max(boxes, key=lambda t: t[2])
            threshold = largest_obj_index[2] * threshold_scalar

            # Reinitialize multi-tracker
            multiTracker = get_multitracker()

            # Run re-detection on current frame
            dif = detect_in_frame(frame, model_path, config_path)

            # Calculate centroids for detected and tracked bounding boxes
            detected_centroids = calc_detected_centroids(dif)
            tracked_centroids = calc_tracked_centroids(boxes)

            # Find the nearest neighbor for each element in the detected set from the tracked set
            nearest = find_nearest(detected_centroids, tracked_centroids)

            # Iterate through the nearest neighbors list
            for i in range(len(nearest[0])):
                # If the nearest neighbor is within the threshold and the number of detected objects has not
                # surpassed the list of tracked objects, associate the two
                if nearest[0][i][0] <= threshold and i < len(old_colors):
                    print("Re-identified an object")
                    colors.append(old_colors[i])
                # Otherwise, assume there is a newly detected object
                else:
                    print("Detected a new object")
                    colors.append(create_color())

            # Convert the bounding box coordinates
            coords = convert_box_coords(dif)

            # Account for any missing colors
            while len(colors) < len(coords):
                colors.append(create_color())

            # Add a tracker for each object
            for bbox in coords:
                multiTracker.add(create_tracker(trackerType), frame, tuple(bbox))

            # Draw the bounding boxes on the frame
            mu = draw_bounding_boxes(frame, coords, colors)

            # Write the frame
            vid_write.write(mu)

        # Otherwise, continue tracking sequence
        else:
            # Update the positions of all tracked objects
            success, boxes = multiTracker.update(frame)

            # Draw the bounding boxes on the frame
            mu = draw_bounding_boxes(frame, boxes, colors)

            # Write the frame
            vid_write.write(mu)

        # Check for cancellation
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print("Frame " + str(count) + " done.")
        count += 1
