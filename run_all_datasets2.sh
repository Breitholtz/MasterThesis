#!/usr/bin/env bash
 
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
#scenes=("fire" "office" "pumpkin" "redkitchen" "stairs")
#percentages=(10 20 30 40 50 60 70 80 90)
percentages=(1 2 5 10 20 30 40 50)
set -x
for percent in "${percentages[@]}"
do
echo ${percent}	
for scene in "${scenes[@]}"
do
  python 3D_overlap2.py --dataset 7Scenes --scene ${scene} --weights logs/7Scenes_${scene}_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar --config_file configs/mapnet.ini --val --percent ${percent}
done
echo "DONE WITH ALL DATASETS"
done

set +x
