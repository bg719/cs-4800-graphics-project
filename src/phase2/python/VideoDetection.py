from imageai.Detection.Custom import CustomVideoObjectDetection


camera = "mygeneratedvideo.mp4"

detector = CustomVideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("SecondModel\\detection_model-ex-030--loss-0018.300.h5")
detector.setJsonPath("SecondModel\\detection_config.json")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=camera, output_file_path="NewModelSinglePool", frames_per_second=20, log_progress=True, minimum_percentage_probability=30, display_object_name=False, display_percentage_probability=False)
