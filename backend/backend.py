import os
import numpy as np
import cv2
import io
import torch
from PIL import Image
from flask import Flask, render_template, url_for, request, send_file, jsonify, send_from_directory

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# import app for DB security reasons
from app import app

# Configs for Mask R-CNN architecture
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("snake_train", )
cfg.DATASETS.TEST = ("snake_test", )
cfg.DATALOADER.NUM_WORKERS = 0                  
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001                      
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.MAX_ITER = 1500                      
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             
cfg.OUTPUT_DIR = "./model/output"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_v3.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.DEVICE = 'cpu'                  
snake_metadata = MetadataCatalog.get("snake_test")
predictor = DefaultPredictor(cfg)

# MySQL connection

# Version 1 REST API endpoints
# image inference endpoint
@app.route('/inference', methods=['POST'])
def inference():
    # handle POST req
    for filename in request.files.keys():
        # get file
        data = request.files[filename]
        img = Image.open(request.files[filename])
        img = np.array(img)
        img = cv2.cvtColor(np.array(img), cv2.IMREAD_COLOR)
        prediction = predictor(img)

        # get boxed snakes
        v = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=1)
        v = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        output = v.get_image()[:, :, ::-1]

        # send the output image back to 
        file_obj = io.BytesIO()
        ret_img = Image.fromarray(output.astype('uint8'))
        ret_img.save(file_obj, 'jpeg')
        file_obj.seek(0)
        return send_file(file_obj, attachment_filename='ret.jpg', mimetype='image/jpeg')

    # TODO: Raise 500 error
    abort(500)

# GPU inference endpoint
@app.route('/gpu', methods=['GET'])
def gpu():
    return jsonify( {'enabled' : torch.cuda.is_available()} )

# training images endpoint
@app.route('/train', methods=['GET'])
def train():
    try:
        return send_from_directory(app.config["TRAIN"], filename="train.zip", mimetype='zip', as_attachment=True)
    except FileNotFoundError:
        # TODO: Raise 500 error
        abort(500)

# test images endpoint
@app.route('/test', methods=['GET'])
def test():
    try:
        return send_from_directory(app.config["TEST"], filename="test.zip", mimetype='zip', as_attachment=True)
    except FileNotFoundError:
        # TODO: Raise 500 error
        abort(500)

# run in Docker container
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
