# Colin Henson and Bryce George
# CS 4800 Computer Graphics
# Finds bowls in a frame using houghTransform from OpenCV

import sys
import cv2 as cv
import numpy as np
import time


def main(argv):
    # Sets a timer to start
    start = time.time()
    # Frame from the video
    default_file = "frame2657.jpg"

    # Opens the image and stores it in src variable
    src = cv.imread(default_file, cv.IMREAD_COLOR)

    # Checks to see if frame was found
    if src is None:
        print("Error")
        return -1
    # Change image to grey scale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Blurs the image
    gray = cv.medianBlur(gray, 5)

    # Sets rows to the shape of the grayscaled image
    rows = gray.shape[0]
    # rows/20, param1=150, param2=12, minRadius=1, maxRadius=30
    # Checks for circles, the parameters can be changed to change sensitivity of algorithm
    # Link to houghcircles documentation below
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 20, param1=150, param2=12, minRadius=1, maxRadius=30)
    count = 0
    # Checks to make sure it found circles
    if circles is not None:
        # Rounds the numbers in the circles list
        # cv.circle cannot take floats
        circles = np.uint16(np.around(circles))

        # Visualizes found circles and their centers on image
        for i in circles[0, :]:
            # Finds the center of the circles detected
            center = (i[0], i[1])
            cv.circle(src, center, 1, (0, 100, 100), 3)

            # Puts a circle around the bowl
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            count += 1

    # Shows the results
    cv.imshow("detected circles", src)
    finish = time.time()
    # Calculates total time
    total = finish - start
    # Prints number of circles found and the time it took to find them
    # There are 46 bowls in this frame
    print("Number of bowls found: {}".format(count))
    print("Total execution time: {}".format(total))
    # Waits until user is done
    cv.waitKey(0)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
