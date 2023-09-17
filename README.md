# Entangled
Computer vision tool for detecting plant entanglement of float plane rudders

<h4>Step one:<h4>
  Set up a conda enviroment with Entangled.yml file
  e.g. : conda env create -f Entangled.yml<br>
  
  <br>If anything fails, like Detection2, create a new conda environment and run the following three lines or select as nessesary:

  <br>pip install 'git+https://github.com/facebookresearch/detectron2.git'<br>
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
  <br>pip install Pillow numpy progress

<h4>Step two:<h4>
  Clone this reposetory and add a directory inside containing your input images (e.g. 'Input_data') and and empty output directory (e.g. 'Entangle_Detect_Out') 
  Also, clone the Cococareader reposetory into the Entangled repo folder as follows:
  git clone git@github.com:jrsykes/CocoaReader.git
<h4>Step three:<h4>
  Add the .pth files, which contain the respective Resnet18 weights and Faster R-CNN weights, to this directory. These are avalable upon request.

<h4>Step four:<h4>
  Run the following commad, eddited acordingly:
  python Entangled.py --images_dir 'Input_data' --out_path 'Entangle_Detect_Out' --Binary_classifier_model_path 'Entangled18.pth' --FaterRCNN_model_path 'FasterRCNN_Rudder.pth'<br>
  
<br>Once this completes, the input images will have been sorted into three directories in the output folder that you created. These will be named 'Entangled', 'NotEntangled' and 'Submerged'.
Images in the submerged class contain those images for which the Faster R-CNN model failed to locate the rudder in the image and so it is assumed to be submerged.
