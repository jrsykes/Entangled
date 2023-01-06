#%%

from detectron2.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
from PIL import Image
import numpy as np
from detectron2 import model_zoo
import torch
import shutil
import torch
from torchvision import transforms
from progress.bar import Bar
import argparse

# %% Define paths
# images_dir = '/local/scratch/jrs596/dat/ElodeaProject/Elodea/Summer2022_Photos/BradGalenaCub_1/DCIM/100GOPRO'
# out_path = "/local/scratch/jrs596/dat/ElodeaProject/Entangle_Detect_Out"


# Binary_classifier_model_path = '/local/scratch/jrs596/dat/ElodeaProject/models/Entangled18.pth'
# FaterRCNN_model_path = '/local/scratch/jrs596/dat/ElodeaProject/FasterRCNN_output/model_final.pth'

#add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', type=str, help='path to images')
parser.add_argument('--out_path', type=str, help='path to output')
parser.add_argument('--Binary_classifier_model_path', type=str, help='path to binary classifier model')
parser.add_argument('--FaterRCNN_model_path', type=str, help='path to FasterRCNN model')

args = parser.parse_args()

#if cuda is available, use gpu otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load FasterRCNN model and custom weights

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  
cfg.MODEL.WEIGHTS = args.FaterRCNN_model_path  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.TEST.DETECTIONS_PER_IMAGE = 1
FasterRCNN = DefaultPredictor(cfg)

#%%Define the plant not plant filter
def plant_notplant_filter(input_img, binary_classifier_model, img_dir, file_name):

	outputs = binary_classifier_model(input_img)
	_, preds = torch.max(outputs, 1) 
		
	# [0][1] = Entangled
	# [0][0] = NotEntangled	
	source = os.path.join(img_dir, file_name)

	if preds.item() == 0:
		dest = os.path.join(args.out_path, 'Entangled')
		os.makedirs(dest, exist_ok=True)

	#If model predicts "not plant"
	elif preds.item() == 1: 
		dest = os.path.join(args.out_path, 'NotEntangled')
		os.makedirs(dest, exist_ok=True)
	
	shutil.copy(source, dest + '/' + file_name)

#%%Crop the bounding box and run the "plant not plant" model

def FasterRCNN_Predict(image_file_name, image_directory, binary_classifier_model):
	
	image = Image.open(os.path.join(image_directory, image_file_name))

	convert_tensor = transforms.ToTensor()
	image_tensor = convert_tensor(image)
	
	image_np = np.asarray(image)
	outputs = FasterRCNN(image_np)


	#If fasterRCNN fails to predicts bounding box, assume the rudder is submerged
	pred_box = (outputs['instances'].pred_boxes).__dict__['tensor'].tolist()
	if len(pred_box) != 0:
		y, x, w, h = int(pred_box[0][0]), int(pred_box[0][1]), int(pred_box[0][2]), int(pred_box[0][3])

		crop_img = image_tensor[:, x:h, y:w].unsqueeze(0).to(device)	

		plant_notplant_filter(input_img = crop_img, binary_classifier_model=binary_classifier_model, img_dir=image_directory, file_name= image_file_name)

	else:
		dest = os.path.join(args.out_path, 'Submerged')
		source = os.path.join(image_directory, image_file_name)
		os.makedirs(dest, exist_ok=True)
		shutil.copy(source, dest + '/' + image_file_name)

#%%Load the "plant not plant" model
Binary_classifier_model = torch.load(args.Binary_classifier_model_path)

input_images = os.listdir(args.images_dir)
with Bar('Disentangling...', max=len(input_images)) as bar:
	for image in input_images:
		FasterRCNN_Predict(image_file_name=image, image_directory=args.images_dir, binary_classifier_model=Binary_classifier_model)
		bar.next()  


#%%