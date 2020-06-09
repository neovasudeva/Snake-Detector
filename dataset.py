import os
import numpy as np
import json
import cv2
import tqdm
import random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# global constants and functions that are platform depenedent
LOAD_DATASET = True
COLAB = False
LOCAL = True
platform = "./" if LOCAL else "/content/drive/My Drive/Colab Notebooks/Snake-Detector/"
def cv2_show_img(title, img):
  """ 
  A wrapper for platform dependent cv2.imshow function
  """
  if COLAB:
    cv2_imshow(img)
  else:
    cv2.imshow(title, img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  

# global constants for labels
labels_dict = {0: "snake", 1: "background"}
HEIGHT = 385
WIDTH = 510
CHANNELS = 3

# function to properly create a Detectron2 Dataset
def detectron2_dataset(path):
    """
    Returns a list[dict] to load into a standard Dataset as specified by 
    https://detectron2.readthedocs.io/tutorials/datasets.html
    
    Params: 
        path = path to directory of images and labels 
        (aka train or test directory)
    """

    # get JSON annotations and images and create list to be returned
    json_files = os.listdir(path + "labels/") 
    image_files = os.listdir(path + "images/")
    std_list = []
   
    # iterate through annotation files
    for json_file in json_files:
        # create dict for image
        image = {}

        # get image name (will be used as image id)
        image_id = json_file[ : json_file.find(".json")]
        image_name = image_id + ".jpg"

        # open JSON file
        annotations = None
        with open(path + "labels/" + json_file) as f:
            annotations = json.load(f)

        # load in standard parameters
        image["file_name"] = path + "images/" + image_name
        image["height"] = annotations["imageHeight"]
        image["width"] = annotations["imageWidth"]
        image["image_id"] = image_id
        
        # load in annotations per shape instance, format of list[dict]
        objs = []
        for shape in annotations["shapes"]:
            # create dict for each box/shape
            obj = {}

            # get polygon and point information
            polygon = shape["points"]
            px = [point[0] for point in polygon]
            py = [point[1] for point in polygon]

            # load fields
            obj["bbox"] = [min(px), min(py), max(px), max(py)]
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["category_id"] = 0
            obj["segmentation"] = [[axis for point in polygon for axis in point]]

            # append obj
            objs.append(obj)

        # add "annotations" field to image
        image["annotations"] = objs
        std_list.append(image)

    # return the std_list
    return std_list


# function to verify if DatasetCatalog is loaded correctly
def test_dataset(path):
    """
    Test if detectron2_dataset function is correct.

    Params:
        path = path to directory of images and labels 
        (aka train or test directory)
    """
    # get train/test type from path
    img_type = "train" if path.find("train") != -1 else "test"

    # get dataset and load metadata
    data = detectron2_dataset(path)
    DatasetCatalog.register("snake_" + img_type, lambda d=img_type: detectron2_dataset(platform + img_type + "/"))
    MetadataCatalog.get("snake_" + img_type).set(thing_classes=["snake"])
    snake_metadata = MetadataCatalog.get("snake_" + img_type)

    # get 3 random images and check if they're correct
    for d in data:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.7)
        vis = visualizer.draw_dataset_dict(d)
        cv2_show_img("Sneks", vis.get_image()[:, :, ::-1])

#detectron2_dataset(platform + "test/")
test_dataset(platform + "train/")
