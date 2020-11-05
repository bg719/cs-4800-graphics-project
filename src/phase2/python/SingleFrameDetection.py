from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-015--loss-0015.368.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="TestFramesKeep\\frame0.jpg", output_image_path="frame456_detected.jpg", )
for detection in detections:
  print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])