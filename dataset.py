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

# function to pull JSON data from annotations
def get_snake_data(path):
    """
    Returns a dictionary of the form { filename, x, y, x2, y2, id, label }

    Params:
        path = string path to json file
    """
    # get path to annotations JSON file
    with open(path) as f:
        full_annots = json.load(f)

    # create dictionary specifically for annotations
    annotations = full_annots["annotations"]

    # return annotations
    return annotations

# function for testing whether get_snake_data function works
def test_json(annotations, path):
    """
    Tests get_snake_data() by iterating over bounding boxes in test
    images
    
    Params:
        annotations = dictionary of images and bounding box locations
        path = path to directory of images
    """
    for filename in annotations:
        # open image
        img = cv2.imread(path + str(filename))

        # get dimensions
        height, width, channels = img.shape

        # draw bounding boxes on img
        for box in annotations[filename]:
            # get coordinates
            x1 = int(box["x"] * width)
            x2 = int(box["x2"] * width)
            y1 = int(box["y"] * height)
            y2 = int(box["y2"] * height)

            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # show image
        cv2_show_img("Snek", img)

# function to properly create a Detectron2 Dataset
def detectron2_dataset(annotations, path):
    """
    Returns a list[dict] to load into a standard Dataset as specified by 
    https://detectron2.readthedocs.io/tutorials/datasets.html
    
    Params: 
        annotations = raw dictionary of images and bounding box locations 
        path = path to directory of images
    """
    # create list to return
    std_list = []

    # iterate through all images and add a new dict for each image
    for id, filename in enumerate(annotations):
        # create dict for image and open file
        img_dict = {}
        
        # get train/test type from path
        img_type = "train/" if path.find("train") != -1 else "test/"

        # load parameters
        try:
            img_dict["file_name"] = platform + img_type + "labeled/" + filename
            img_dict["height"] = cv2.imread(path + filename).shape[:1][0]
            img_dict["width"] = cv2.imread(path + filename).shape[1:2][0]
            img_dict["image_id"] = id
        except AttributeError:
            print("img type: ", img_type, ", path: ", path)

        # load "annotations" parameter, comes in the form of list[dict]
        # NOTE: box is a dictionary containing bounding box data for an image
        objs = []
        for box in annotations[filename]:
            # create dict for box
            obj = {}

            # load in box coordinates and mode
            obj["bbox"] = [int(box["x"]*WIDTH), int(box["y"]*HEIGHT), int(box["x2"]*WIDTH), int(box["y2"]*HEIGHT)]
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["category_id"] = 0

            # add dictionary to objs
            objs.append(obj)

        img_dict["annotations"] = objs

        # append img dict to the list of images
        std_list.append(img_dict)

    # return standard list
    return std_list

# function to verify if DatasetCatalog is loaded correctly
def test_dataset(path):
    """
    Test if detectron2_dataset function is correct.

    Params:
        path = path to directory of images
    """
    # get train/test type from path
    img_type = "train" if path.find("train") != -1 else "test"

    # get dataset and load metadata
    annotation = get_snake_data(platform + img_type + "/_annotations.json")
    data = detectron2_dataset(annotation, path)
    snake_metadata = MetadataCatalog.get("snake_" + img_type)

    # get 3 random images and check if they're correct
    for d in random.sample(data, 3):
        # get image
        img = cv2.imread(d["file_name"])

        # draw bounding` box with class name and random-colord box 
        if img is not None:
            visualizer = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2_show_img("Snek", vis.get_image()[:, :, ::-1])
        else:
            print("Failure occured while loading images or drawing boxes.")
            break

# Extended trainer class that allows evaluation
class SnakeTrainer(DefaultTrainer):
    """
    Class that implements the build_evaluator function,
    allows for custom evaluation
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(platform + "eval", exist_ok=True)
            output_folder = platform + "eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# get dataset and load into DatasetCatalog (and MetadataCatalog)
#if LOAD_DATASET:
#  for d in ["train", "test"]:
#      annotations = get_snake_data(platform + d + "/_annotations.json")
#      DatasetCatalog.register("snake_" + d, lambda x=d: detectron2_dataset(annotations, platform + d + "/labeled/"))
#      MetadataCatalog.get("snake_" + d).set(thing_classes=["snake"])

# load training dataset
annotations = get_snake_data(platform + "train/_annotations.json")
DatasetCatalog.register("snake_train", lambda d="train": detectron2_dataset(annotations, platform + d + "/labeled/"))
MetadataCatalog.get("snake_train").set(thing_classes=["snake"])

# Configs for Faster R-CNN architecture
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.DATASETS.TRAIN = ("snake_train", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0                  # threads for data loading
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001                      # change me
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.STEPS = (1000, 1500, 2000)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.MAX_ITER = 2000                      # may need to increase later
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # may need to increase later
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 class (snake)

# load outputs into "output" directory
cfg.OUTPUT_DIR = platform + "output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = SnakeTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

