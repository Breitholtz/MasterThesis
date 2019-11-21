#!/usr/bin/env bash
 
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
#scenes=("fire" "office" "pumpkin" "redkitchen" "stairs")
percentages=(10)
#40 50 60 70 80 90)
#percentages=(10 20 30 40 50)
set -x
for percent in "${percentages[@]}"
do
echo ${percent}	
for scene in "${scenes[@]}"
do
  python 3D_overlap_skip.py --dataset 7Scenes --scene ${scene} --weights logs/7Scenes_${scene}_posenet_posenet_learn_beta_logq/epoch_300.pth.tar --config_file configs/posenet.ini --val --percent ${percent} --frames 1
done
echo "DONE WITH ALL DATASETS"
done

set +x
