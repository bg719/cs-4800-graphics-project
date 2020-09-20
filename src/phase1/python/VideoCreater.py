import cv2
import os
from PIL import Image
# Directory of frames
os.chdir("C:\\Users\\Colin Henson\\Desktop\\SupportforProcess\\FirstVid")
path = "C:\\Users\\Colin Henson\\Desktop\\SupportforProcess\\FirstVid"

def generate_video():
    image_folder = '.'  # make sure to use your folder
    video_name = 'secondtestvideo.avi'
    os.chdir("C:\\Users\\Colin Henson\\Desktop\\SupportforProcess\\FirstVid")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if any
    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 3, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

        # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


# Calling the generate_video function
generate_video()