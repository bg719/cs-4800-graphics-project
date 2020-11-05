from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="drive/My Drive/bowls/BowlsModelInfo")
trainer.evaluateModel(model_path="drive/My Drive/bowls/BowlsModelInfo/models", json_path="drive/My Drive/bowls/BowlsModelInfo/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)