import cv2
# Bowls Video File
vidcap = cv2.VideoCapture('Bowls.mp4')
# Checks read of video
success, image = vidcap.read()
# Counter
count = 0
# While there are frames in the video
while success:
    # Write frame with name of number
    cv2.imwrite("frame%d.jpg" % count, image)
    # Checks for next frame
    success, image = vidcap.read()
    print('Read a n ew frame: ', success)
    count += 1


