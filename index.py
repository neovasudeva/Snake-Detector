import os
import numpy as np
import cv2
import io
from PIL import Image
from flask import Flask, render_template, url_for, request, send_file

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Configs for Mask R-CNN architecture
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("snake_train", )
cfg.DATASETS.TEST = ("snake_test", )
cfg.DATALOADER.NUM_WORKERS = 0                  # threads for data loading
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001                      # change me
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.MAX_ITER = 1500                      # may need to increase later
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # may need to increase later
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 class (snake)
cfg.OUTPUT_DIR = "./model/output"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_v2.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.94
cfg.MODEL.DEVICE='cpu'                          
snake_metadata = MetadataCatalog.get("snake_test")
predictor = DefaultPredictor(cfg)

# start application
app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['POST', 'GET'])
def home():
    # handle POST req
    if request.method == 'POST':
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

    # always return template
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

