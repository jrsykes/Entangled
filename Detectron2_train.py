#%%
import sys, os, distutils.core
import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


#%%

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
#%%
from detectron2.data.datasets import register_coco_instances
root = '/local/scratch/jrs596'

train_img_folder= root + '/dat/ElodeaProject/BB4_combined_split/train'
val_img_folder= root + '/dat/ElodeaProject/BB4_combined_split/val'

register_coco_instances("my_dataset_train", {}, train_img_folder + "/annotations/custom_train.json", train_img_folder + "/images")
register_coco_instances("my_dataset_val", {}, val_img_folder + "/annotations/custom_val.json", val_img_folder + "/images")


#%%

from detectron2.engine import DefaultTrainer

print(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 300    
cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.OUTPUT_DIR = root + '/dat/ElodeaProject/FasterRCNN_output'
cfg.SOLVER.MAX_ITER = 1000



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# %%
