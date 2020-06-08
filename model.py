import os
import numpy as np
import json
import cv2
import random

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dataset import platform, cv2_show_img, cfg, get_snake_data, detectron2_dataset

# load test dataset
# NOTE: had some weird problem where it was training on test dataset
annotations = get_snake_data(platform + "test/_annotations.json")
DatasetCatalog.register("snake_test", lambda d="test": detectron2_dataset(annotations, platform + d + "/labeled/"))
MetadataCatalog.get("snake_test").set(thing_classes=["snake"])

# load model and make predictions
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.DATASETS.TEST = ("snake_test", )
snake_metadata = MetadataCatalog.get("snake_test")
predictor = DefaultPredictor(cfg)

# display some test results
annotations = get_snake_data(platform + "test/_annotations.json")
eval_dataset = detectron2_dataset(annotations, platform + "test/labeled/")
for d in eval_dataset:
    img = cv2.imread(d["file_name"])  #cv2.imread(platform + "temp/" + d) 
    outputs = predictor(img)
    
    v = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_show_img("Prediction", v.get_image()[:, :, ::-1])
