# Entangled
Computer vision tool for detecting plant entanglement of float plane rudders

Step one:
  Set up conda enviroment with Entangled.yml file
  e.g. : conda env create -f Entangled.yml

  Alternativley, create a new conda environment and run the following three lines:

  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
  pip install Pillow numpy progress 

Step two:
  Clone this reposetory and add a directory containing your input images (e.g. 'Input_data') and and empty output directory (e.g. Entangle_Detect_Out) 

Step three:
  Add the .pth files, which contain the respective Resnet18 weights and Faster R-CNN weights, to this directory. These are avalable upon request.

Step four:
  Run the following commad, eddited acordingly:
  python Entangled.py --images_dir 'Input_data' --out_path 'Entangle_Detect_Out' --Binary_classifier_model_path 'Entangled18.pth' --FaterRCNN_model_path 'FasterRCNN_Rudder.pth'

Once this completes, the input images will have been sorted into three directories in the output folder that you created. These will be named 'Entangled', 'NotEntangled' and 'Submerged'.
Images in the submerged class contain those when the Faster R-CNN model failed to locate the rudder in the image and so it is assumed to be submerged.
