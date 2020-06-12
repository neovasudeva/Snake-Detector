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

print("Configurations being set...")

# global constants and functions that are platform depenedent
COLAB = False
LOCAL = True
platform = "./" if LOCAL else "/content/drive/My Drive/Colab Notebooks/Snake-Detector-v2/"
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

# load train dataset
LOAD_TRAIN = False
if LOAD_TRAIN:
    DatasetCatalog.register("snake_train", lambda d="train/": detectron2_dataset(platform + d))
    MetadataCatalog.get("snake_train").set(thing_classes=["snake"])
    LOAD_TRAIN = False

# Configs for Mask R-CNN architecture
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("snake_train", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0                  # threads for data loading
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001                     # change me
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.MAX_ITER = 1500                      # may need to increase later
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # may need to increase later
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 class (snake)
cfg.OUTPUT_DIR = platform + "output/"

# RUN INFERENCE ON CPU, CHANGE TO GPU LATER
cfg.MODEL.DEVICE='cpu'                          
print("Configurations are done!")

# load test dataset
# NOTE: had some weird problem where it was training on test dataset
LOAD_TEST = False
if LOAD_TEST:
    DatasetCatalog.register("snake_test", lambda d="test/": detectron2_dataset(platform + d))
    MetadataCatalog.get("snake_test").set(thing_classes=["snake"])

# load model and make predictions
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_v2.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.94
cfg.DATASETS.TEST = ("snake_test", )
snake_metadata = MetadataCatalog.get("snake_test")
predictor = DefaultPredictor(cfg)

# display some test results
#eval_dataset = detectron2_dataset(platform + "test/")
#for d in eval_dataset:
print("Running predictions...")
for file in os.listdir(platform + "test/unlabeled_imgs/"):
    img = cv2.imread(platform + "test/unlabeled_imgs/" + file) #cv2.imread(d["file_name"])  
    outputs = predictor(img)
    
    v = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_show_img("Prediction", v.get_image()[:, :, ::-1])

# running evaluation
#evaluator = COCOEvaluator("snake_test", cfg, False, output_dir=platform + "output/")
#val_loader = build_detection_test_loader(cfg, "snake_test")
#inference_on_dataset(trainer.model, val_loader, evaluator)
