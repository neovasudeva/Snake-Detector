import os
import numpy as np
import json
import cv2
import tqdm
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

# global constants for labels
labels_dict = {0: "snake", 1: "background"}
HEIGHT = 385
WIDTH = 510
CHANNELS = 3

# function to pull JSON data from annotations
def get_snake_data():
    """
    Returns a dictionary of the form { filename, x, y, x2, y2, id, label }
    """
    # get path to annotations JSON file
    json_file = "./training/_annotations.json"
    with open(json_file) as f:
        full_annots = json.load(f)

    # create dictionary specifically for annotations
    annotations = full_annots["annotations"]

    # return annotations
    return annotations

# function for testing whether get_snake_data function works
def test_json(annotations):
    """
    Tests get_snake_data() by iterating over bounding boxes in test
    images
    
    Params:
        annotations = dictionary of images and bounding box locations
    """
    for filename in annotations:
        # open image
        img = cv2.imread("./training/labeled/" + str(filename))

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
def detectron2_dataset(annotations):
    """
    Returns a list[dict] to load into a standard Dataset as specified by 
    https://detectron2.readthedocs.io/tutorials/datasets.html
    
    Params: 
        annotations = raw dictionary of images and bounding box locations 
    """
    # create list to return
    std_list = []

    # iterate through all images and add a new dict for each image
    for id, filename in enumerate(annotations):
        # create dict for image and open file
        img_dict = {}
        
        # load parameters
        img_dict["file_name"] = filename
        img_dict["height"] = cv2.imread("./training/labeled/" + filename).shape[:1][0]
        img_dict["width"] = cv2.imread("./training/labeled/" + filename).shape[1:2][0]
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

# get dataset and load into DatasetCatalog (and MetadataCatalog)
# TODO: create test data set in test folder
annotations = get_snake_data()
for d in ["train", "val"]:
    DatasetCatalog.register("snake_" + d, lambda d=d: detectron2_dataset(annotations))
    MetadataCatalog.get("snake_" + d).set(thing_classes=["snake"])

# check data
data = detectron2_dataset(annotations)
snake_metadata = MetadataCatalog.get("snake_train")
for d in random.sample(data, 3):
    # get image
    img = cv2.imread("./training/labeled/" + d["file_name"])

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




























