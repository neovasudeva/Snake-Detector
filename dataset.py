import os
import numpy as np
import json
import cv2
import tqdm
import random

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

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
        print(height, width, channels)

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
        cv2.imshow("snek", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

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
        
        # load parameters
        img_dict["file_name"] = filename
        img_dict["height"] = cv2.imread(path + filename).shape[:1][0]
        img_dict["width"] = cv2.imread(path + filename).shape[1:2][0]
        img_dict["image_id"] = id

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
    start_idx = [i for i, n in enumerate(path) if n == '/'][0] + 1
    end_idx = [i for i, n in enumerate(path) if n == '/'][1]

    # get dataset and load metadata
    annotation = get_snake_data("./" + path[start_idx : end_idx] + "/_annotations.json")
    data = detectron2_dataset(annotation, path)
    snake_metadata = MetadataCatalog.get("snake_" + path[start_idx : end_idx])

    # get 3 random images and check if they're correct
    for d in random.sample(data, 3):
        # get image
        img = cv2.imread(path + d["file_name"])

        # draw bounding box with class name and random-colord box 
        if img is not None:
            visualizer = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow("idk", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("Failure occured while loading images or drawing boxes.")
            break

# get dataset and load into DatasetCatalog (and MetadataCatalog)
# TODO: create test data set in test folder
for d in ["train", "test"]:
    annotations = get_snake_data("./" + d + "/_annotations.json")
    DatasetCatalog.register("snake_" + d, lambda d=d: detectron2_dataset(annotations, "./" + d + "/labeled/"))
    MetadataCatalog.get("snake_" + d).set(thing_classes=["snake"])

# check data
# test_dataset("./train/labeled/")

# Configs for Mask R-CNN X101-FPN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("snake_train", )
cfg.DATASETS.TEST = ("snake_test", )
cfg.DATALOADER.NUM_WORKERS = 4                  # threads for data loading
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0005                     # change me
cfg.SOLVER.MAX_ITER = 500                       # may need to increase later
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # may need to increase later
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 class (snake)

# load outputs into "output" directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
print("Training is starting...")
trainer.train()
print("Training is finished!")

# save the model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "snake_detector.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)











