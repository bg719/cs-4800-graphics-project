import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("frame2657.jpg")

b_f_im = cv.medianBlur(img, 5)

edges = cv.Canny(b_f_im, 100, 200, 3)

# cv.imshow("Just Canny", edges)


contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

contour_list = []

for contour in contours:
    approx = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
    area = cv.contourArea(contour)
    if(len(approx) > 2) & (len(approx) < 30) & (area > 10):
        contour_list.append(contour)

cv.drawContours(img, contour_list, -1, (255, 0, 0), 2)
cv.imshow("Full detection", img)
cv.waitKey(0)


plt.show()

