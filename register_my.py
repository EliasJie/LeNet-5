# -*- coding: utf-8 -*-

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import random
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances('self_coco_train', {},
                        './my_coco_dataset/data_dataset_coco_train/annotations.json',
                       './my_coco_dataset/data_dataset_coco_train')
register_coco_instances('self_coco_val', {},
                        './my_coco_dataset/data_dataset_coco_val/annotations.json',
                       './my_coco_dataset/data_dataset_coco_val')

coco_val_metadata = MetadataCatalog.get("self_coco_val")
dataset_dicts = DatasetCatalog.get("self_coco_val")
coco_train_metadata = MetadataCatalog.get("self_coco_train")
dataset_dicts1 = DatasetCatalog.get("self_coco_train")
coco_val_metadata
coco_train_metadata

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("self_coco_train")
dataset_dicts = DatasetCatalog.get("self_coco_train")

import random

#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("self_coco_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 400000 // 4000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("self_coco_val", )
cfg.MODEL.DEVICE = 0

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=True)
trainer.train()

# predictor = DefaultPredictor(cfg)
# from detectron2.utils.visualizer import ColorMode
#
# def get_file_names(folder_path):
#     file_names = []
#     for file_name in os.listdir(folder_path):
#         if os.path.isfile(os.path.join(folder_path, file_name)):
#             file_names.append(file_name)
#     return file_names
#
# folder_path = './test_img/img'  # 替换为你的文件夹路径
# file_names = get_file_names(folder_path)
# # print(file_names)
# from detectron2.utils.visualizer import Visualizer
# import random
#
#
# for d in random.sample(file_names, 10):
#     print(f"{folder_path}/{d}")
#     im = cv2.imread(f"{folder_path}/{d}")
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=coco_val_metadata,
#                    scale=0.8,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     image = v.get_image()[:, :, ::-1]  # 获取图像
#     cv2.imwrite(f"./test_img/res/{d}", image)


import os

print(cfg.MODEL.WEIGHTS)
os.system("python ./tools/deploy/export_model.py --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml     --output ./output_cpu --export-method tracing --format torchscript     MODEL.WEIGHTS ./output/model_final.pth  MODEL.DEVICE cpu")