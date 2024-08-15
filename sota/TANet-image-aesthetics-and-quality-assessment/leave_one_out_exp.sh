#!/bin/bash

# dataname="spaq"
dataname="eva"
gpu_id=1

cd code/TAD66K

# # nni experiment, find best hyperparameters
# # After 20 trials, the best hyperparameters:
# # batch_size:40
# # init_lr_res365_last:0.000001
# # init_lr_mobileNet:0.000001
# # init_lr_head:1e-7
# # init_lr_head_rgb:1e-7
# # init_lr_hypernet:3e-7
# # init_lr_tygertnet:3e-7
# # init_lr:0.000001
# for folder in data_json/for_leave_one_out/$dataname/*/; do
#   echo "Processing $folder"
#   if [ -d "$folder" ]; then
#     folder_name=$(basename "$folder")
#     train_json_path="$folder/train.json"
#     test_json_path="$folder/test.json"

#     if [ -f "$train_json_path" ] && [ -f "$test_json_path" ]; then
#       log_dir="exp_log/$dataname/$folder_name"
#       mkdir -p $log_dir

#       dst_yml="temp_config.yml"
#       cp "config.yml" "$dst_yml"

#       train_cmd="CUDA_VISIBLE_DEVICES=$gpu_id python train_nni.py --path_to_save_csv $folder --experiment_dir_name $log_dir"
#       yq eval ".trial.command = \"$train_cmd\"" -i "$dst_yml"

#       nnictl create --config $dst_yml -p 8998
      
#       echo "[DONE] $train_cmd"
#       break # only run one experiment
#     else
#       echo "Warning: $train_json_path or $test_json_path does not exist."
#     fi
#   fi
# done


for folder in data_json/for_leave_one_out/$dataname/*/; do
  echo "Processing $folder"
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    train_json_path="$folder/train.json"
    test_json_path="$folder/test.json"

    if [ -f "$train_json_path" ] && [ -f "$test_json_path" ]; then
      log_dir="exp_log/$dataname/$folder_name"
      mkdir -p $log_dir

      CUDA_VISIBLE_DEVICES=$gpu_id python -u train_nni.py --path_to_images data/ --path_to_save_csv $folder --experiment_dir_name $log_dir \
                                          --batch_size 40 --init_lr_res365_last 0.000001 --init_lr_mobileNet 0.000001 --init_lr_head 1e-7 --init_lr_head_rgb 1e-7 \
                                          --init_lr_hypernet 3e-7 --init_lr_tygertnet 3e-7 --init_lr 0.000001 \
                                          > $log_dir/log.txt
      
      echo "Training done for $folder"
    else
      echo "Warning: $folder does not contain train.json or test.json"
    fi
  fi
done