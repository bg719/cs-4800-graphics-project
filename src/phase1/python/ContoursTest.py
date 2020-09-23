import cv2 as cv
import numpy as np

im = cv.imread("frame2657.jpg")
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 150, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
drawing = cv.drawContours(im, contours, -1, (0,255,0), 3)
cv.imshow("something", drawing)
cv.waitKey(0)