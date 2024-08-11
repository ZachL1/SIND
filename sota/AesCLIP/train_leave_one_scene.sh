#!/bin/bash

dataname="eva"
gpu_id=1


for folder in data_json/for_leave_one_out/$dataname/*/; do
  echo "Processing $folder"
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    train_json_path="$folder/train.json"
    test_json_path="$folder/test.json"

    if [ -f "$train_json_path" ] && [ -f "$test_json_path" ]; then
      log_dir="exp_log/$dataname/$folder_name"
      mkdir -p $log_dir
      CUDA_VISIBLE_DEVICES=$gpu_id python train/leave_one_scene_train.py --path_to_images data/ --path_to_save_csv $folder --experiment_dir_name $log_dir
      
      echo "Training done for $folder"
    else
      echo "Warning: $folder does not contain train.json or test.json"
    fi
  fi
done