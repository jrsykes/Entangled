source activate convnext

python '/home/userfs/j/jrs596/scripts/CocoaReader/DisNet/Torch_Custom_CNNs_wandbSweep.py' \
        --model_name 'Entangled18' \
        --project_name "Elodea" \
        --root '/local/scratch/jrs596/dat/ElodeaProject' \
        --data_dir 'FasterRCNN_output/Rudders_split' \
        --min_epochs 10 \
        --initial_batch_size 32 \
        --patience 30 \
        --min_batch_size 4 \
        --custom_pretrained \
        --custom_pretrained_weights '/local/scratch/jrs596/dat/models/PlantNotPlant_SemiSup.pkl'