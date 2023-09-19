source activate Entangled

python '/home/userfs/j/jrs596/scripts/Entangled/Torch_Custom_CNNs2.2.1.py' \
        --model_name 'Entangled18' \
        --project_name "Elodea" \
        --entity 'frankslab' \
        --root '/local/scratch/jrs596/dat/ElodeaProject' \
        --data_dir 'FasterRCNN_output/Rudders_split' \
        --arch 'resnet18' \
        --input_size 224 \
        --min_epochs 10 \
        --max_epochs 100 \
        --batch_size 32 \
        --patience 20 \
        --weights '/local/scratch/jrs596/dat/models/PlantNotPlant_SemiSup.pkl'