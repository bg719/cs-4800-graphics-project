import sys
import cv2 as cv
import numpy as np
import time


def main(argv):
    # Sets a timer to start
    start = time.time()
    default_file = "frame2657.jpg"

    src = cv.imread(default_file, cv.IMREAD_COLOR)

    if src is None:
        print("Error")
        return -1
    # Change image to grey scale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Blurs the image
    gray = cv.medianBlur(gray, 5)
    # gray = cv.GaussianBlur(gray, (7, 7),sigmaX=1.5,sigmaY=1.5)
    # Sets rows to the shape of the grayscaled image
    rows = gray.shape[0]
    # rows/20, param1=150, param2=12, minRadius=1, maxRadius=30
    # Checks for circles, the parameters can be changed to change sensitivity of algorithm
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 20, param1=150, param2=12, minRadius=1, maxRadius=30)
    count = 0

    if circles is not None:
        # Rounds the numbers in the circles list
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # Finds the center of the circles detected
            count += 1
            center = (i[0], i[1])
            cv.circle(src, center, 1, (0, 100, 100), 3)

            # Puts a circle around the bowl
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    # Shows the results
    cv.imshow("detected circles", src)
    finish = time.time()
    total = finish - start
    # Prints number of circles found and the time it took to find them
    # There are 46 bowls in this frame
    print(count)
    print(total)
    cv.waitKey(0)
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
